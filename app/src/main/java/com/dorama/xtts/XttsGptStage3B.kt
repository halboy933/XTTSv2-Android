package com.dorama.xtts

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.charset.StandardCharsets
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max

class XttsGptStage3B(private val filesDir: File) {
 private val modelRoot=File(filesDir,"xtts_models")
 private val env=OrtEnvironment.getEnvironment()

 data class NpyFloat(val shape:IntArray,val data:FloatArray)
 data class BpeResult(val cleaned:String,val ids:IntArray)

 fun run(text:String,progress:(String)->Unit):String {
  val totalStart=System.nanoTime()
  val cache=File(filesDir,"conditioning_cache/cond_latents.f32")
  require(cache.exists()) { "Conditioning cache missing. Run Stage 3A first." }

  progress("Stage 3B 1/6: loading tokenizer + conditioning cache…")
  val cond=readFloat32LE(cache)
  require(cond.size==32*1024) { "cond_latents cache expected 32768 floats, got ${cond.size}" }

  val vocabFile=findRequired("vocab.json")
  val tokenizer=XttsBpeTokenizer(vocabFile)
  val bpe=tokenizer.encodeRussian(text)
  require(bpe.ids.isNotEmpty()) { "Tokenizer returned no tokens" }
  require(bpe.ids.size+2<=404) { "Text is too long for text_pos_embedding (${bpe.ids.size} BPE tokens)" }

  progress("Stage 3B 2/6: loading XTTS embedding tables…")
  val tEmb=System.nanoTime()
  val textEmb=readNpyFloat(findRequired("text_embedding.npy"))
  val textPos=readNpyFloat(findRequired("text_pos_embedding.npy"))
  val melEmb=readNpyFloat(findRequired("mel_embedding.npy"))
  val melPos=readNpyFloat(findRequired("mel_pos_embedding.npy"))
  val embMs=elapsedMs(tEmb)
  validate2d(textEmb,"text_embedding",1024)
  validate2d(textPos,"text_pos_embedding",1024)
  validate2d(melEmb,"mel_embedding",1024)
  validate2d(melPos,"mel_pos_embedding",1024)

  val metadata=JSONObject(findRequired("metadata.json").readText())
  val nLayer=metadata.getInt("n_layer")
  val numHeads=metadata.getInt("num_heads")
  val headDim=metadata.getInt("head_dim")
  val startAudio=metadata.getInt("start_audio_token")
  val stopAudio=metadata.getInt("stop_audio_token")
  val startText=tokenizer.tokenId("[START]")
  val stopText=tokenizer.tokenId("[STOP]")

  require(nLayer==30) { "Expected 30 GPT layers, got $nLayer" }
  require(numHeads*headDim==1024) { "GPT head dimensions do not equal 1024" }
  require(startAudio in 0 until melEmb.shape[0]) { "start_audio_token out of range" }

  progress("Stage 3B 3/6: building prefix embeddings…")
  val tPrefix=System.nanoTime()
  val textTokens=IntArray(bpe.ids.size+2)
  textTokens[0]=startText
  System.arraycopy(bpe.ids,0,textTokens,1,bpe.ids.size)
  textTokens[textTokens.lastIndex]=stopText
  val prefixLen=32+textTokens.size+1
  val prefix=FloatArray(prefixLen*1024)
  System.arraycopy(cond,0,prefix,0,cond.size)
  var row=32
  for(i in textTokens.indices) {
   val token=textTokens[i]
   require(token in 0 until textEmb.shape[0]) { "Text token $token outside text_embedding" }
   require(i<textPos.shape[0]) { "Text position $i outside text_pos_embedding" }
   val dst=row*1024
   val te=token*1024
   val tp=i*1024
   for(j in 0 until 1024) prefix[dst+j]=textEmb.data[te+j]+textPos.data[tp+j]
   row++
  }
  val audioDst=row*1024
  val ae=startAudio*1024
  for(j in 0 until 1024) prefix[audioDst+j]=melEmb.data[ae+j]+melPos.data[j]
  val prefixMs=elapsedMs(tPrefix)

  progress("Stage 3B 4/6: loading gpt_model_int8.onnx…")
  val gptModel=findRequired("gpt_model_int8.onnx")
  val opts=OrtSession.SessionOptions()
  val loadStart=System.nanoTime()
  val session=env.createSession(gptModel.absolutePath,opts)
  val loadMs=elapsedMs(loadStart)

  try {
   progress("Stage 3B 5/6: GPT prefill (${prefixLen} positions)…")
   val prefillStart=System.nanoTime()
   val firstInputs=linkedMapOf<String,OnnxTensor>()
   firstInputs["inputs_embeds"]=tensor(prefix,longArrayOf(1,prefixLen.toLong(),1024))
   firstInputs["attention_mask"]=tensor(FloatArray(prefixLen){1f},longArrayOf(1,prefixLen.toLong()))
   for(i in 0 until nLayer) {
    firstInputs["past_key_$i"]=tensor(FloatArray(0),longArrayOf(1,numHeads.toLong(),0,headDim.toLong()))
    firstInputs["past_value_$i"]=tensor(FloatArray(0),longArrayOf(1,numHeads.toLong(),0,headDim.toLong()))
   }

   val prefill:OrtSession.Result
   try {
    prefill=session.run(firstInputs)
   } finally {
    firstInputs.values.forEach { runCatching { it.close() } }
   }
   val prefillMs=elapsedMs(prefillStart)

   prefill.use { p ->
    val logits=p[0] as OnnxTensor
    val hidden=p[1] as OnnxTensor
    val logitsShape=(logits.info as TensorInfo).shape
    val hiddenShape=(hidden.info as TensorInfo).shape
    require(logitsShape.contentEquals(longArrayOf(1,prefixLen.toLong(),1026))) { "Unexpected logits shape ${shape(logitsShape)}" }
    require(hiddenShape.contentEquals(longArrayOf(1,prefixLen.toLong(),1024))) { "Unexpected hidden_states shape ${shape(hiddenShape)}" }

    val firstToken=argmaxLastRow(logits,1026)
    val kv0=(p[2] as OnnxTensor).info as TensorInfo
    val expectedKv=longArrayOf(1,numHeads.toLong(),prefixLen.toLong(),headDim.toLong())
    require(kv0.shape.contentEquals(expectedKv)) { "Unexpected KV-cache shape ${shape(kv0.shape)}" }
    val hiddenMean=meanAbsLastRow(hidden,1024)

    var decodeMs=0L
    var secondToken=-1
    var decodeLogitsShape=longArrayOf()
    var decodeKvShape=longArrayOf()

    if(firstToken!=stopAudio) {
     progress("Stage 3B 6/6: first KV-cache decode step…")
     require(firstToken in 0 until melEmb.shape[0]) { "First GPT token $firstToken outside mel_embedding" }
     require(melPos.shape[0]>1) { "mel_pos_embedding has no position 1" }
     val nextEmb=FloatArray(1024)
     val me=firstToken*1024
     val mp=1024
     for(j in 0 until 1024) nextEmb[j]=melEmb.data[me+j]+melPos.data[mp+j]

     val secondInputs=linkedMapOf<String,OnnxTensor>()
     val owned=ArrayList<OnnxTensor>()
     val nextTensor=tensor(nextEmb,longArrayOf(1,1,1024)); owned.add(nextTensor); secondInputs["inputs_embeds"]=nextTensor
     val maskTensor=tensor(FloatArray(prefixLen+1){1f},longArrayOf(1,(prefixLen+1).toLong())); owned.add(maskTensor); secondInputs["attention_mask"]=maskTensor
     for(i in 0 until nLayer) {
      secondInputs["past_key_$i"]=p[2+2*i] as OnnxTensor
      secondInputs["past_value_$i"]=p[2+2*i+1] as OnnxTensor
     }

     val tDecode=System.nanoTime()
     val second:OrtSession.Result
     try {
      second=session.run(secondInputs)
     } finally {
      owned.forEach { runCatching { it.close() } }
     }
     decodeMs=elapsedMs(tDecode)
     second.use { s ->
      val logits2=s[0] as OnnxTensor
      secondToken=argmaxLastRow(logits2,1026)
      decodeLogitsShape=((logits2.info as TensorInfo).shape).clone()
      decodeKvShape=(((s[2] as OnnxTensor).info as TensorInfo).shape).clone()
      require(decodeLogitsShape.contentEquals(longArrayOf(1,1,1026))) { "Unexpected decode logits ${shape(decodeLogitsShape)}" }
      require(decodeKvShape.contentEquals(longArrayOf(1,numHeads.toLong(),(prefixLen+1).toLong(),headDim.toLong()))) {
       "Unexpected decode KV ${shape(decodeKvShape)}"
      }
     }
    }

    val totalMs=elapsedMs(totalStart)
    return buildString {
     append("STAGE 3B SUCCESS\n\n")
     append("Clean Russian: ${bpe.cleaned}\n")
     append("BPE tokens: ${bpe.ids.size}\n")
     append("Token IDs: ${bpe.ids.joinToString(", ")}\n")
     append("Text sentinels: START=$startText STOP=$stopText\n")
     append("Audio sentinels: START=$startAudio STOP=$stopAudio\n\n")
     append("Embeddings loaded: ${embMs} ms\n")
     append("Prefix: [1, $prefixLen, 1024] (${prefixMs} ms)\n")
     append("GPT INT8 load: ${loadMs} ms\n")
     append("Prefill logits: ${shape(logitsShape)} (${prefillMs} ms)\n")
     append("Prefill hidden: ${shape(hiddenShape)} meanAbs=${"%.6f".format(Locale.US,hiddenMean)}\n")
     append("KV[0] after prefill: ${shape(kv0.shape)}\n")
     append("First greedy audio token: $firstToken\n")
     if(firstToken==stopAudio) {
      append("Model emitted STOP on first token; decode step skipped.\n")
     } else {
      append("Decode logits: ${shape(decodeLogitsShape)} (${decodeMs} ms)\n")
      append("KV[0] after decode: ${shape(decodeKvShape)}\n")
      append("Second greedy candidate: $secondToken\n")
     }
     append("\nTotal Stage 3B: ${totalMs} ms\n")
     append("Ready for Stage 3C: sampling loop + HiFi-GAN → WAV.")
    }
   }
  } finally {
   session.close()
   opts.close()
  }
 }

 private fun tensor(data:FloatArray,shape:LongArray)=OnnxTensor.createTensor(env,FloatBuffer.wrap(data),shape)

 private fun argmaxLastRow(t:OnnxTensor,width:Int):Int {
  val info=t.info as TensorInfo
  val rows=(info.shape.fold(1L){a,b->a*max(1L,b)} / width).toInt()
  val fb=t.floatBuffer
  require(fb!=null && fb.capacity()>=rows*width) { "Cannot read GPT logits" }
  val offset=(rows-1)*width
  var best=0
  var bestV=fb.get(offset)
  for(i in 1 until width) {
   val v=fb.get(offset+i)
   if(v>bestV) { bestV=v; best=i }
  }
  return best
 }

 private fun meanAbsLastRow(t:OnnxTensor,width:Int):Double {
  val info=t.info as TensorInfo
  val rows=(info.shape.fold(1L){a,b->a*max(1L,b)} / width).toInt()
  val fb=t.floatBuffer ?: error("Cannot read hidden_states")
  val offset=(rows-1)*width
  var sum=0.0
  for(i in 0 until width) sum+=abs(fb.get(offset+i).toDouble())
  return sum/width
 }

 private fun validate2d(a:NpyFloat,name:String,width:Int) {
  require(a.shape.size==2 && a.shape[1]==width) { "$name expected [N,$width], got ${a.shape.contentToString()}" }
  require(a.data.size==a.shape[0]*a.shape[1]) { "$name data length mismatch" }
 }

 private fun findRequired(name:String):File {
  val f=modelRoot.walkTopDown().firstOrNull { it.isFile && it.name==name }
  require(f!=null) { "$name not found. Import model ZIP first." }
  return f
 }

 private fun readFloat32LE(file:File):FloatArray {
  val b=file.readBytes()
  require(b.size%4==0) { "Invalid float32 file: ${file.name}" }
  val bb=ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
  return FloatArray(b.size/4) { bb.float }
 }

 private fun readNpyFloat(file:File):NpyFloat {
  val b=file.readBytes()
  require(b.size>=12) { "Invalid NPY ${file.name}" }
  require((b[0].toInt() and 0xff)==0x93 && String(b,1,5,StandardCharsets.US_ASCII)=="NUMPY") { "Invalid NPY magic: ${file.name}" }
  val major=b[6].toInt() and 0xff
  val headerStart:Int
  val headerLen:Int
  if(major==1) {
   headerStart=10
   headerLen=(b[8].toInt() and 0xff) or ((b[9].toInt() and 0xff) shl 8)
  } else {
   headerStart=12
   headerLen=(b[8].toInt() and 0xff) or ((b[9].toInt() and 0xff) shl 8) or ((b[10].toInt() and 0xff) shl 16) or ((b[11].toInt() and 0xff) shl 24)
  }
  val dataOffset=headerStart+headerLen
  require(dataOffset<=b.size) { "Invalid NPY header: ${file.name}" }
  val header=String(b,headerStart,headerLen,StandardCharsets.US_ASCII)
  require(header.contains("f4")) { "Expected float32 NPY: ${file.name}" }
  require(!header.contains("'fortran_order': True") && !header.contains("\"fortran_order\": true")) { "Fortran NPY unsupported" }
  val m=Regex("shape['\\\"]?\\s*:\\s*\\(([^)]*)\\)").find(header)
   ?: Regex("'shape'\\s*:\\s*\\(([^)]*)\\)").find(header)
   ?: error("Cannot parse NPY shape: ${file.name}")
  val shape=m.groupValues[1].split(',').mapNotNull { it.trim().takeIf(String::isNotEmpty)?.toIntOrNull() }.toIntArray()
  require(shape.isNotEmpty()) { "Empty NPY shape: ${file.name}" }
  val count=shape.fold(1L){a,v->a*v}.toInt()
  require(dataOffset+count*4<=b.size) { "NPY data truncated: ${file.name}" }
  val bb=ByteBuffer.wrap(b,dataOffset,count*4).order(ByteOrder.LITTLE_ENDIAN)
  val data=FloatArray(count) { bb.float }
  return NpyFloat(shape,data)
 }

 private fun elapsedMs(start:Long)=(System.nanoTime()-start)/1_000_000L
 private fun shape(s:LongArray)="[${s.joinToString(", ")}]"
}

internal class XttsBpeTokenizer(file:File) {
 private val root=JSONObject(file.readText())
 private val model=root.getJSONObject("model")
 private val vocab=HashMap<String,Int>()
 private val added=HashMap<String,Int>()
 private val mergeRank=HashMap<String,Int>()
 private val unkId:Int
 private val specials:List<String>

 init {
  val v=model.getJSONObject("vocab")
  val keys=v.keys()
  while(keys.hasNext()) {
   val k=keys.next()
   vocab[k]=v.getInt(k)
  }
  val a=root.optJSONArray("added_tokens") ?: JSONArray()
  for(i in 0 until a.length()) {
   val o=a.getJSONObject(i)
   added[o.getString("content")]=o.getInt("id")
  }
  val merges=model.optJSONArray("merges") ?: JSONArray()
  for(i in 0 until merges.length()) {
   val item=merges.get(i)
   val pair=when(item) {
    is String -> item.trim().split(Regex("\\s+"),limit=2)
    is JSONArray -> listOf(item.getString(0),item.getString(1))
    else -> emptyList()
   }
   if(pair.size==2) mergeRank[pairKey(pair[0],pair[1])]=i
  }
  unkId=vocab[model.optString("unk_token","[UNK]")] ?: vocab["[UNK]"] ?: 1
  specials=(added.keys+listOf("[ru]","[SPACE]","[START]","[STOP]")).distinct().sortedByDescending { it.length }
 }

 fun tokenId(token:String)=added[token] ?: vocab[token] ?: error("Tokenizer token missing: $token")

 fun encodeRussian(source:String):XttsGptStage3B.BpeResult {
  val cleaned=cleanRussian(source)
  require(cleaned.isNotBlank()) { "Text is empty after normalization" }
  val prepared="[ru]"+cleaned.replace(" ","[SPACE]")
  val ids=ArrayList<Int>()
  var pos=0
  while(pos<prepared.length) {
   val special=specials.firstOrNull { prepared.startsWith(it,pos) }
   if(special!=null) {
    ids.add(tokenId(special)); pos+=special.length; continue
   }
   var next=prepared.length
   for(sp in specials) {
    val at=prepared.indexOf(sp,pos)
    if(at>=0 && at<next) next=at
   }
   tokenizePlain(prepared.substring(pos,next),ids)
   pos=next
  }
  return XttsGptStage3B.BpeResult(cleaned,ids.toIntArray())
 }

 private fun tokenizePlain(chunk:String,out:MutableList<Int>) {
  if(chunk.isEmpty()) return
  val re=Regex("[\\p{L}\\p{N}_]+|[^\\p{L}\\p{N}_\\s]+")
  for(m in re.findAll(chunk)) {
   val piece=m.value
   if(piece.isNotEmpty()) for(sym in bpe(piece)) out.add(vocab[sym] ?: unkId)
  }
 }

 private fun bpe(token:String):List<String> {
  val symbols=token.codePoints().toArray().map { String(Character.toChars(it)) }.toMutableList()
  if(symbols.size<2) return symbols
  while(symbols.size>=2) {
   var bestRank=Int.MAX_VALUE
   var bestA:String?=null
   var bestB:String?=null
   for(i in 0 until symbols.size-1) {
    val r=mergeRank[pairKey(symbols[i],symbols[i+1])] ?: continue
    if(r<bestRank) { bestRank=r; bestA=symbols[i]; bestB=symbols[i+1] }
   }
   if(bestA==null || bestB==null) break
   val merged=ArrayList<String>(symbols.size)
   var i=0
   while(i<symbols.size) {
    if(i<symbols.size-1 && symbols[i]==bestA && symbols[i+1]==bestB) {
     merged.add(bestA+bestB); i+=2
    } else { merged.add(symbols[i]); i++ }
   }
   symbols.clear(); symbols.addAll(merged)
  }
  return symbols
 }

 private fun pairKey(a:String,b:String)=a+'\u0001'+b

 private fun cleanRussian(input:String):String {
  var s=input.replace("\"","").lowercase(Locale.forLanguageTag("ru"))
  s=s.replace(Regex("(?i)\\bг-жа\\b"),"госпожа")
   .replace(Regex("(?i)\\bг-н\\b"),"господин")
   .replace(Regex("(?i)\\bд-р\\b"),"доктор")
  s=s.replace("&"," и ").replace("@"," собака ").replace("%"," процентов ")
   .replace("#"," номер ").replace("$"," доллар ").replace("£"," фунт ").replace("°"," градус ")
  s=Regex("\\d+").replace(s) { m -> numberToRussian(m.value.toLongOrNull() ?: 0L) }
  return s.replace(Regex("\\s+")," ").trim()
 }

 private fun numberToRussian(n:Long):String {
  if(n==0L) return "ноль"
  if(n<0) return "минус "+numberToRussian(-n)
  if(n>999_999_999L) return n.toString()
  val parts=ArrayList<String>()
  fun under1000(x:Int,feminine:Boolean=false):String {
   val o=if(feminine) arrayOf("","одна","две","три","четыре","пять","шесть","семь","восемь","девять") else arrayOf("","один","два","три","четыре","пять","шесть","семь","восемь","девять")
   val teens=arrayOf("десять","одиннадцать","двенадцать","тринадцать","четырнадцать","пятнадцать","шестнадцать","семнадцать","восемнадцать","девятнадцать")
   val tens=arrayOf("","","двадцать","тридцать","сорок","пятьдесят","шестьдесят","семьдесят","восемьдесят","девяносто")
   val hundreds=arrayOf("","сто","двести","триста","четыреста","пятьсот","шестьсот","семьсот","восемьсот","девятьсот")
   val a=ArrayList<String>(); var v=x
   if(v>=100){a.add(hundreds[v/100]);v%=100}
   if(v in 10..19){a.add(teens[v-10]);v=0}else if(v>=20){a.add(tens[v/10]);v%=10}
   if(v>0)a.add(o[v]); return a.joinToString(" ")
  }
  fun form(v:Int,one:String,few:String,many:String):String {
   val v100=v%100; val v10=v%10
   return if(v100 in 11..14) many else when(v10){1->one;2,3,4->few;else->many}
  }
  val millions=(n/1_000_000).toInt()
  val thousands=((n/1000)%1000).toInt()
  val rest=(n%1000).toInt()
  if(millions>0){parts.add(under1000(millions));parts.add(form(millions,"миллион","миллиона","миллионов"))}
  if(thousands>0){parts.add(under1000(thousands,true));parts.add(form(thousands,"тысяча","тысячи","тысяч"))}
  if(rest>0)parts.add(under1000(rest))
  return parts.joinToString(" ")
 }
}
