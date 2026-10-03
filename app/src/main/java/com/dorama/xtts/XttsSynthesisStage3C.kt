package com.dorama.xtts

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.charset.StandardCharsets
import java.util.Locale
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.random.Random

class XttsSynthesisStage3C(private val filesDir: File) {
 private val modelRoot=File(filesDir,"xtts_models")
 private val cacheRoot=File(filesDir,"conditioning_cache")
 private val env=OrtEnvironment.getEnvironment()

 data class NpyFloat(val shape:IntArray,val data:FloatArray)
 data class Sampling(
  val temperature:Float=0.75f,
  val topK:Int=50,
  val topP:Float=0.85f,
  val repetitionPenalty:Float=10.0f,
  val seed:Int?=null
 )

 fun run(text:String,progress:(String)->Unit):String =
  run(text,Sampling(),"xtts_generated.wav",progress)

 fun run(
  text:String,
  sampling:Sampling,
  outputName:String="xtts_generated.wav",
  progress:(String)->Unit
 ):String {
  require(outputName.matches(Regex("[A-Za-z0-9._-]+"))) { "Invalid output file name" }
  val totalStart=System.nanoTime()
  val condFile=File(cacheRoot,"cond_latents.f32")
  val speakerFile=File(cacheRoot,"speaker_embedding.f32")
  require(condFile.exists() && speakerFile.exists()) {
   "Conditioning cache incomplete. Run Stage 3A first."
  }

  progress("Stage 3C 1/7: loading voice cache + Russian tokenizer…")
  val cond=readFloat32LE(condFile)
  val speaker=readFloat32LE(speakerFile)
  require(cond.size==32*1024) { "cond_latents cache invalid: ${cond.size} floats" }
  require(speaker.size==512) { "speaker_embedding cache invalid: ${speaker.size} floats" }

  val tokenizer=XttsBpeTokenizer(findRequired("vocab.json"))
  val bpe=tokenizer.encodeRussian(text)
  require(bpe.ids.isNotEmpty()) { "Tokenizer returned no tokens" }
  require(bpe.ids.size+2<=404) { "Text is too long: ${bpe.ids.size} BPE tokens" }

  progress("Stage 3C 2/7: loading embeddings…")
  val embStart=System.nanoTime()
  val textEmb=readNpyFloat(findRequired("text_embedding.npy"))
  val textPos=readNpyFloat(findRequired("text_pos_embedding.npy"))
  val melEmb=readNpyFloat(findRequired("mel_embedding.npy"))
  val melPos=readNpyFloat(findRequired("mel_pos_embedding.npy"))
  val embMs=elapsedMs(embStart)
  validate2d(textEmb,"text_embedding",1024)
  validate2d(textPos,"text_pos_embedding",1024)
  validate2d(melEmb,"mel_embedding",1024)
  validate2d(melPos,"mel_pos_embedding",1024)

  val meta=JSONObject(findRequired("metadata.json").readText())
  val nLayer=meta.getInt("n_layer")
  val numHeads=meta.getInt("num_heads")
  val headDim=meta.getInt("head_dim")
  val startAudio=meta.getInt("start_audio_token")
  val stopAudio=meta.getInt("stop_audio_token")
  val startText=tokenizer.tokenId("[START]")
  val stopText=tokenizer.tokenId("[STOP]")
  require(nLayer==30 && numHeads==16 && headDim==64) {
   "Unexpected GPT config: layers=$nLayer heads=$numHeads headDim=$headDim"
  }

  progress("Stage 3C 3/7: building GPT prefix…")
  val prefixStart=System.nanoTime()
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
  val prefixMs=elapsedMs(prefixStart)

  progress("Stage 3C 4/7: loading GPT INT8…")
  val opts=OrtSession.SessionOptions()
  val loadStart=System.nanoTime()
  val gpt=env.createSession(findRequired("gpt_model_int8.onnx").absolutePath,opts)
  val gptLoadMs=elapsedMs(loadStart)

  val generated=ArrayList<Int>()
  val latents=ArrayList<FloatArray>()
  val used=BooleanArray(1026)
  used[1]=true
  val rng=if(sampling.seed!=null) Random(sampling.seed) else Random.Default
  val maxTokens=min(605,melPos.shape[0]-1)

  var current:OrtSession.Result?=null
  var gptRunMs=0L
  var stoppedByToken=false

  try {
   progress("Stage 3C 5/7: GPT prefill…")
   val firstInputs=linkedMapOf<String,OnnxTensor>()
   firstInputs["inputs_embeds"]=tensor(prefix,longArrayOf(1,prefixLen.toLong(),1024))
   firstInputs["attention_mask"]=tensor(FloatArray(prefixLen){1f},longArrayOf(1,prefixLen.toLong()))
   for(i in 0 until nLayer) {
    firstInputs["past_key_$i"]=tensor(FloatArray(0),longArrayOf(1,numHeads.toLong(),0,headDim.toLong()))
    firstInputs["past_value_$i"]=tensor(FloatArray(0),longArrayOf(1,numHeads.toLong(),0,headDim.toLong()))
   }
   val preStart=System.nanoTime()
   try {
    current=gpt.run(firstInputs)
   } finally {
    firstInputs.values.forEach { runCatching { it.close() } }
   }
   gptRunMs+=elapsedMs(preStart)

   progress("Stage 3C 6/7: generating audio tokens…")
   while(generated.size<maxTokens) {
    val result=current ?: error("GPT result missing")
    val logits=result[0] as OnnxTensor
    val hidden=result[1] as OnnxTensor

    val scores=lastRow(logits,1026)
    val token=sampleToken(scores,used,sampling,rng)

    if(token==stopAudio) {
     stoppedByToken=true
     break
    }
    require(token in 0 until melEmb.shape[0]) { "GPT emitted invalid audio token $token" }

    latents.add(lastRow(hidden,1024))
    generated.add(token)
    used[token]=true

    if(generated.size%10==0) {
     progress("Stage 3C 6/7: generated ${generated.size} audio tokens…")
    }
    if(generated.size>=maxTokens) break

    val posIndex=generated.size
    require(posIndex<melPos.shape[0]) { "mel_pos_embedding exhausted at $posIndex" }
    val nextEmb=FloatArray(1024)
    val me=token*1024
    val mp=posIndex*1024
    for(j in 0 until 1024) nextEmb[j]=melEmb.data[me+j]+melPos.data[mp+j]

    val nextInputs=linkedMapOf<String,OnnxTensor>()
    val owned=ArrayList<OnnxTensor>()
    val e=tensor(nextEmb,longArrayOf(1,1,1024))
    val mask=tensor(FloatArray(prefixLen+generated.size){1f},longArrayOf(1,(prefixLen+generated.size).toLong()))
    owned.add(e); owned.add(mask)
    nextInputs["inputs_embeds"]=e
    nextInputs["attention_mask"]=mask
    for(i in 0 until nLayer) {
     nextInputs["past_key_$i"]=result[2+2*i] as OnnxTensor
     nextInputs["past_value_$i"]=result[2+2*i+1] as OnnxTensor
    }

    val stepStart=System.nanoTime()
    val next:OrtSession.Result
    try {
     next=gpt.run(nextInputs)
    } finally {
     owned.forEach { runCatching { it.close() } }
    }
    gptRunMs+=elapsedMs(stepStart)
    result.close()
    current=next
   }
  } finally {
   current?.let { runCatching { it.close() } }
   gpt.close()
   opts.close()
  }

  require(latents.isNotEmpty()) { "GPT produced no audio latents" }

  progress("Stage 3C 7/7: HiFi-GAN vocoder → WAV…")
  val latentFlat=FloatArray(latents.size*1024)
  for(i in latents.indices) System.arraycopy(latents[i],0,latentFlat,i*1024,1024)

  val vocoderOpts=OrtSession.SessionOptions()
  val vocoderLoadStart=System.nanoTime()
  val vocoder=env.createSession(findRequired("hifigan_vocoder.onnx").absolutePath,vocoderOpts)
  val vocoderLoadMs=elapsedMs(vocoderLoadStart)
  val vocoderStart=System.nanoTime()
  val vocoderInputs=linkedMapOf<String,OnnxTensor>()
  vocoderInputs["latents"]=tensor(latentFlat,longArrayOf(1,latents.size.toLong(),1024))
  vocoderInputs["speaker_embedding"]=tensor(speaker,longArrayOf(1,512,1))
  val audio:FloatArray
  try {
   vocoder.run(vocoderInputs).use { out ->
    val t=out[0] as OnnxTensor
    val outShape=(t.info as TensorInfo).shape
    require(outShape.size==3 && outShape[0]==1L && outShape[1]==1L && outShape[2]>0) {
     "Unexpected vocoder output ${shape(outShape)}"
    }
    val fb=t.floatBuffer ?: error("Cannot read vocoder audio")
    audio=FloatArray(fb.remaining())
    fb.get(audio)
   }
  } finally {
   vocoderInputs.values.forEach { runCatching { it.close() } }
   vocoder.close()
   vocoderOpts.close()
  }
  val vocoderMs=elapsedMs(vocoderStart)
  require(audio.isNotEmpty() && audio.all { it.isFinite() }) { "Vocoder returned invalid audio" }

  val outFile=File(filesDir,outputName)
  writeWav16(outFile,audio,24000)
  val peak=audio.maxOf { abs(it) }
  val seconds=audio.size/24000.0
  val totalMs=elapsedMs(totalStart)

  return buildString {
   append("STAGE 3C SUCCESS\n\n")
   append("Text: ${bpe.cleaned}\n")
   append("BPE tokens: ${bpe.ids.size}\n")
   append("Prefix: [1, $prefixLen, 1024] (${prefixMs} ms)\n")
   append("Embeddings load: ${embMs} ms\n")
   append("GPT INT8 load: ${gptLoadMs} ms\n")
   append("Sampling: temp=${sampling.temperature}, topK=${sampling.topK}, topP=${sampling.topP}, rep=${sampling.repetitionPenalty}\n")
   append("Seed: ${sampling.seed?.toString() ?: "random"}\n")
   append("Generated audio tokens: ${generated.size}\n")
   append("Stop token reached: $stoppedByToken\n")
   append("GPT inference total: ${gptRunMs} ms\n")
   append("HiFi-GAN load: ${vocoderLoadMs} ms\n")
   append("HiFi-GAN inference: ${vocoderMs} ms\n")
   append("Audio: ${audio.size} samples @ 24000 Hz = ${"%.2f".format(Locale.US,seconds)} s\n")
   append("Peak abs: ${"%.4f".format(Locale.US,peak)}\n")
   append("Saved: ${outFile.name}\n")
   append("Total Stage 3C: ${totalMs} ms\n\n")
   append("Use PLAY GENERATED WAV to hear the cloned voice.")
  }
 }

 private fun sampleToken(raw:FloatArray,used:BooleanArray,cfg:Sampling,rng:Random):Int {
  val scores=raw.copyOf()
  for(i in scores.indices) {
   if(i<used.size && used[i]) {
    scores[i]=if(scores[i]<0f) scores[i]*cfg.repetitionPenalty else scores[i]/cfg.repetitionPenalty
   }
   scores[i]/=cfg.temperature
  }

  if(cfg.topK>0 && cfg.topK<scores.size) {
   val sorted=scores.copyOf().also { it.sort() }
   val threshold=sorted[sorted.size-cfg.topK]
   for(i in scores.indices) if(scores[i]<threshold) scores[i]=Float.NEGATIVE_INFINITY
  }

  if(cfg.topP<1f) {
   val order=scores.indices.sortedByDescending { scores[it] }
   var maxScore=Float.NEGATIVE_INFINITY
   for(i in order) if(scores[i]>maxScore) maxScore=scores[i]
   val weights=DoubleArray(order.size)
   var sum=0.0
   for(k in order.indices) {
    val v=if(scores[order[k]].isFinite()) exp((scores[order[k]]-maxScore).toDouble()) else 0.0
    weights[k]=v
    sum+=v
   }
   var cum=0.0
   for(k in order.indices) {
    val p=if(sum>0) weights[k]/sum else 0.0
    if(cum>cfg.topP) scores[order[k]]=Float.NEGATIVE_INFINITY
    cum+=p
   }
  }

  var maxScore=Float.NEGATIVE_INFINITY
  for(v in scores) if(v.isFinite() && v>maxScore) maxScore=v
  require(maxScore.isFinite()) { "All GPT logits were filtered out" }

  val probs=DoubleArray(scores.size)
  var total=0.0
  for(i in scores.indices) {
   val p=if(scores[i].isFinite()) exp((scores[i]-maxScore).toDouble()) else 0.0
   probs[i]=p
   total+=p
  }
  require(total>0.0 && total.isFinite()) { "Invalid GPT probability distribution" }
  var r=rng.nextDouble()*total
  for(i in probs.indices) {
   r-=probs[i]
   if(r<=0.0) return i
  }
  return probs.lastIndex
 }

 private fun lastRow(t:OnnxTensor,width:Int):FloatArray {
  val info=t.info as TensorInfo
  val elements=info.shape.fold(1L){a,b->a*max(1L,b)}
  require(elements%width==0L) { "Tensor width mismatch: ${shape(info.shape)} / $width" }
  val rows=(elements/width).toInt()
  val fb=t.floatBuffer ?: error("Cannot read tensor")
  require(fb.capacity()>=rows*width) { "Tensor buffer too small" }
  val offset=(rows-1)*width
  return FloatArray(width) { fb.get(offset+it) }
 }

 private fun tensor(data:FloatArray,shape:LongArray)=
  OnnxTensor.createTensor(env,FloatBuffer.wrap(data),shape)

 private fun validate2d(a:NpyFloat,name:String,width:Int) {
  require(a.shape.size==2 && a.shape[1]==width) {
   "$name expected [N,$width], got ${a.shape.contentToString()}"
  }
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
  require((b[0].toInt() and 0xff)==0x93 && String(b,1,5,StandardCharsets.US_ASCII)=="NUMPY") {
   "Invalid NPY magic: ${file.name}"
  }
  val major=b[6].toInt() and 0xff
  val headerStart:Int
  val headerLen:Int
  if(major==1) {
   headerStart=10
   headerLen=(b[8].toInt() and 0xff) or ((b[9].toInt() and 0xff) shl 8)
  } else {
   headerStart=12
   headerLen=(b[8].toInt() and 0xff) or ((b[9].toInt() and 0xff) shl 8) or
    ((b[10].toInt() and 0xff) shl 16) or ((b[11].toInt() and 0xff) shl 24)
  }
  val dataOffset=headerStart+headerLen
  require(dataOffset<=b.size) { "Invalid NPY header: ${file.name}" }
  val header=String(b,headerStart,headerLen,StandardCharsets.US_ASCII)
  require(header.contains("f4")) { "Expected float32 NPY: ${file.name}" }
  require(!header.contains("'fortran_order': True") && !header.contains("\"fortran_order\": true")) {
   "Fortran NPY unsupported"
  }
  val m=Regex("shape['\\\"]?\\s*:\\s*\\(([^)]*)\\)").find(header)
   ?: Regex("'shape'\\s*:\\s*\\(([^)]*)\\)").find(header)
   ?: error("Cannot parse NPY shape: ${file.name}")
  val shape=m.groupValues[1].split(',').mapNotNull {
   it.trim().takeIf(String::isNotEmpty)?.toIntOrNull()
  }.toIntArray()
  val count=shape.fold(1L){a,v->a*v}.toInt()
  require(dataOffset+count*4<=b.size) { "NPY data truncated: ${file.name}" }
  val bb=ByteBuffer.wrap(b,dataOffset,count*4).order(ByteOrder.LITTLE_ENDIAN)
  return NpyFloat(shape,FloatArray(count){bb.float})
 }

 private fun writeWav16(file:File,audio:FloatArray,sampleRate:Int) {
  val dataBytes=audio.size*2
  RandomAccessFile(file,"rw").use { f ->
   f.setLength(0)
   f.writeBytes("RIFF")
   writeIntLE(f,36+dataBytes)
   f.writeBytes("WAVE")
   f.writeBytes("fmt ")
   writeIntLE(f,16)
   writeShortLE(f,1)
   writeShortLE(f,1)
   writeIntLE(f,sampleRate)
   writeIntLE(f,sampleRate*2)
   writeShortLE(f,2)
   writeShortLE(f,16)
   f.writeBytes("data")
   writeIntLE(f,dataBytes)
   for(v in audio) {
    val s=(v.coerceIn(-1f,1f)*32767f).roundToInt()
    writeShortLE(f,s)
   }
  }
 }

 private fun writeIntLE(f:RandomAccessFile,v:Int) {
  f.write(v and 0xff); f.write((v ushr 8) and 0xff); f.write((v ushr 16) and 0xff); f.write((v ushr 24) and 0xff)
 }
 private fun writeShortLE(f:RandomAccessFile,v:Int) {
  f.write(v and 0xff); f.write((v ushr 8) and 0xff)
 }

 private fun elapsedMs(start:Long)=(System.nanoTime()-start)/1_000_000L
 private fun shape(s:LongArray)="[${s.joinToString(", ")}]"
}
