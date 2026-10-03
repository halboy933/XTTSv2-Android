package com.dorama.xtts

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

class XttsReferenceLab(private val filesDir:File) {
 data class Candidate(
  val index:Int,
  val startSec:Double,
  val durationSec:Double,
  val file:File,
  val centroidSimilarity:Double,
  val speechRatio:Double
 )

 data class Result(
  val all:List<Candidate>,
  val top5:List<Candidate>,
  val report:String
 )

 private data class WavData(val samples:FloatArray,val sampleRate:Int)
 private data class Prepared(
  val index:Int,
  val startSec:Double,
  val durationSec:Double,
  val file:File,
  val speechRatio:Double
 )

 fun run(longReference:File,progress:(String)->Unit):Result {
  require(longReference.exists()) { "reference_long.wav is missing." }

  // Playback/Save/Share resolve candidate names directly under filesDir.
  // Remove stale root-level copies before a new Reference Lab run.
  filesDir.listFiles()
   ?.filter { it.isFile && it.name.matches(Regex("xtts_refcand_\\d{2}\\.wav")) }
   ?.forEach { runCatching { it.delete() } }

  progress("Stage 3M 1/5: читаю длинную запись…")
  val wav=readWav(longReference)
  val totalSec=wav.samples.size.toDouble()/wav.sampleRate
  require(totalSec>=12.0) {
   "Для Reference Lab нужно хотя бы 12 секунд речи; сейчас %.2f с".format(Locale.US,totalSec)
  }

  val windowSec=min(5.5,totalSec/2.0)
  val windowSamples=(windowSec*wav.sampleRate).roundToInt()
  val maxStart=(wav.samples.size-windowSamples).coerceAtLeast(0)
  val requested=18
  val outDir=File(filesDir,"reference_lab_candidates")
  outDir.deleteRecursively()
  outDir.mkdirs()

  progress("Stage 3M 2/5: готовлю $requested перекрывающихся окон…")
  val prepared=ArrayList<Prepared>()

  for(i in 0 until requested) {
   val start=if(maxStart==0) 0 else ((maxStart.toLong()*i)/(requested-1)).toInt()
   val raw=wav.samples.copyOfRange(start,(start+windowSamples).coerceAtMost(wav.samples.size))
   val trimmed=trimEdges(raw,wav.sampleRate)
   if(trimmed.size < (3.2*wav.sampleRate).roundToInt()) continue

   val speechRatio=activeRatio(trimmed,wav.sampleRate)
   if(speechRatio<0.48) continue

   val normalized=normalizePeak(trimmed,0.75f)
   val file=File(outDir,"xtts_refcand_${String.format(Locale.US,"%02d",i+1)}.wav")
   writePcm16(file,normalized,wav.sampleRate)

   prepared.add(
    Prepared(
     index=i+1,
     startSec=start.toDouble()/wav.sampleRate,
     durationSec=normalized.size.toDouble()/wav.sampleRate,
     file=file,
     speechRatio=speechRatio
    )
   )
  }

  require(prepared.size>=5) {
   "После фильтра осталось только ${prepared.size} пригодных окон. Нужна более непрерывная чистая запись."
  }

  progress("Stage 3M 3/5: считаю speaker embedding для ${prepared.size} окон…")
  val conditioning=XttsConditioning(filesDir)
  val embeddings=prepared.mapIndexed { i,item ->
   progress(
    "Stage 3M 3/5: embedding ${i+1}/${prepared.size}\n"+
     "окно ${item.index} • старт ${String.format(Locale.US,"%.1f",item.startSec)} с"
   )
   conditioning.computeSpeakerEmbedding(item.file)
  }

  progress("Stage 3M 4/5: строю центр реального голоса…")
  val centroid=centroid(embeddings)

  val candidates=prepared.indices.map { i ->
   val p=prepared[i]
   Candidate(
    index=p.index,
    startSec=p.startSec,
    durationSec=p.durationSec,
    file=p.file,
    centroidSimilarity=cosine(centroid,embeddings[i]),
    speechRatio=p.speechRatio
   )
  }

  val ranked=candidates.sortedWith(
   compareByDescending<Candidate> { it.centroidSimilarity }
    .thenByDescending { it.speechRatio }
  )
  val top5=ranked.take(5)

  // Analysis candidates live in reference_lab_candidates/.
  // UI playback/save/share look in filesDir, so keep playable Top-5 copies there.
  val playableTop5=top5.map { candidate ->
   val playable=File(filesDir,candidate.file.name)
   candidate.file.copyTo(playable,overwrite=true)
   candidate.copy(file=playable)
  }

  progress("Stage 3M 5/5: готово — Top 5 сохранены для прослушивания.")

  val report=buildString {
   append("STAGE 3M SUCCESS — Reference Lab\n\n")
   append("Source duration: ${String.format(Locale.US,"%.2f",totalSec)} s\n")
   append("Windows requested: $requested\n")
   append("Windows accepted: ${candidates.size}\n")
   append("Processing: trim edge silence + normalize peak to 0.75\n")
   append("Current voice profile: NOT CHANGED\n\n")
   append("TOP 5 REAL REFERENCE WINDOWS:\n")
   for((rank,c) in playableTop5.withIndex()) {
    append("${rank+1}. Ref ${c.index}")
    append(" • start ${String.format(Locale.US,"%.1f",c.startSec)} s")
    append(" • dur ${String.format(Locale.US,"%.2f",c.durationSec)} s")
    append(" • speaker ${String.format(Locale.US,"%.4f",c.centroidSimilarity)}")
    append(" • speech ${String.format(Locale.US,"%.0f",c.speechRatio*100)}%")
    append("\n")
   }
   append("\nПрослушай Top 5 и отметь 2–3 фрагмента, которые звучат наиболее естественно.")
  }

  return Result(candidates,playableTop5,report)
 }

 private fun centroid(vectors:List<FloatArray>):FloatArray {
  require(vectors.isNotEmpty())
  val out=FloatArray(512)
  for(v in vectors) {
   require(v.size==512)
   var normSq=0.0
   for(x in v) normSq+=x.toDouble()*x
   val norm=sqrt(normSq).coerceAtLeast(1e-12)
   for(i in out.indices) out[i]+=(v[i]/norm).toFloat()
  }
  var normSq=0.0
  for(x in out) normSq+=x.toDouble()*x
  val norm=sqrt(normSq).coerceAtLeast(1e-12)
  for(i in out.indices) out[i]=(out[i]/norm).toFloat()
  return out
 }

 private fun cosine(a:FloatArray,b:FloatArray):Double {
  var dot=0.0; var aa=0.0; var bb=0.0
  for(i in a.indices) {
   val x=a[i].toDouble(); val y=b[i].toDouble()
   dot+=x*y; aa+=x*x; bb+=y*y
  }
  return dot/(sqrt(aa).coerceAtLeast(1e-12)*sqrt(bb).coerceAtLeast(1e-12))
 }

 private fun rms(samples:FloatArray):Double {
  var e=0.0
  for(v in samples) e+=v.toDouble()*v
  return sqrt(e/samples.size.coerceAtLeast(1))
 }

 private fun trimEdges(samples:FloatArray,sampleRate:Int):FloatArray {
  if(samples.isEmpty()) return samples
  val block=(sampleRate*0.020).roundToInt().coerceAtLeast(1)
  val threshold=max(0.006,rms(samples)*0.28)

  var first=-1
  var last=-1
  var blockIndex=0
  var start=0
  while(start<samples.size) {
   val end=min(samples.size,start+block)
   var e=0.0
   for(i in start until end) e+=samples[i].toDouble()*samples[i]
   val brms=sqrt(e/(end-start).coerceAtLeast(1))
   if(brms>=threshold) {
    if(first<0) first=blockIndex
    last=blockIndex
   }
   blockIndex++
   start=end
  }
  if(first<0 || last<0) return samples.copyOf()

  val pad=(sampleRate*0.080).roundToInt()
  val from=(first*block-pad).coerceAtLeast(0)
  val to=((last+1)*block+pad).coerceAtMost(samples.size)
  return samples.copyOfRange(from,to)
 }

 private fun activeRatio(samples:FloatArray,sampleRate:Int):Double {
  val block=(sampleRate*0.020).roundToInt().coerceAtLeast(1)
  val threshold=max(0.006,rms(samples)*0.24)
  var active=0
  var total=0
  var start=0
  while(start<samples.size) {
   val end=min(samples.size,start+block)
   var e=0.0
   for(i in start until end) e+=samples[i].toDouble()*samples[i]
   if(sqrt(e/(end-start).coerceAtLeast(1))>=threshold) active++
   total++
   start=end
  }
  return active.toDouble()/total.coerceAtLeast(1)
 }

 private fun normalizePeak(samples:FloatArray,target:Float):FloatArray {
  var peak=0f
  for(v in samples) peak=max(peak,abs(v))
  if(peak<1e-6f) return samples.copyOf()
  val gain=(target/peak).coerceAtMost(8f)
  return FloatArray(samples.size) { i -> (samples[i]*gain).coerceIn(-1f,1f) }
 }

 private fun readWav(file:File):WavData {
  val b=file.readBytes()
  require(b.size>=44 && ascii(b,0,4)=="RIFF" && ascii(b,8,4)=="WAVE") { "Invalid WAV" }
  var pos=12
  var format=-1; var channels=-1; var sampleRate=-1; var bits=-1; var blockAlign=-1
  var dataOffset=-1; var dataSize=-1

  while(pos+8<=b.size) {
   val id=ascii(b,pos,4)
   val size=leInt(b,pos+4)
   val body=pos+8
   require(size>=0 && body+size<=b.size) { "Invalid WAV chunk" }
   when(id) {
    "fmt " -> {
     format=leShort(b,body)
     channels=leShort(b,body+2)
     sampleRate=leInt(b,body+4)
     blockAlign=leShort(b,body+12)
     bits=leShort(b,body+14)
    }
    "data" -> { dataOffset=body; dataSize=size }
   }
   pos=body+size+(size and 1)
  }

  require(dataOffset>=0 && dataSize>0 && channels>0 && sampleRate>0 && blockAlign>0) { "Invalid WAV data" }
  require(format==1 || format==3) { "Unsupported WAV format $format" }

  val frames=dataSize/blockAlign
  val mono=FloatArray(frames)
  val bytesPerSample=(bits+7)/8

  for(i in 0 until frames) {
   var sum=0.0
   val frameBase=dataOffset+i*blockAlign
   for(ch in 0 until channels) {
    val o=frameBase+ch*bytesPerSample
    val sample=when {
     format==3 && bits==32 -> Float.fromBits(leInt(b,o)).toDouble()
     format==1 && bits==8 -> ((b[o].toInt() and 0xff)-128)/128.0
     format==1 && bits==16 -> leSignedShort(b,o)/32768.0
     format==1 && bits==24 -> leSigned24(b,o)/8388608.0
     format==1 && bits==32 -> leInt(b,o)/2147483648.0
     else -> error("Unsupported WAV: format=$format bits=$bits")
    }
    sum+=sample
   }
   mono[i]=(sum/channels).coerceIn(-1.0,1.0).toFloat()
  }
  return WavData(mono,sampleRate)
 }

 private fun writePcm16(file:File,samples:FloatArray,sampleRate:Int) {
  val dataBytes=samples.size*2
  RandomAccessFile(file,"rw").use { f ->
   f.setLength(0)
   f.writeBytes("RIFF"); writeIntLE(f,36+dataBytes)
   f.writeBytes("WAVE"); f.writeBytes("fmt "); writeIntLE(f,16)
   writeShortLE(f,1); writeShortLE(f,1)
   writeIntLE(f,sampleRate); writeIntLE(f,sampleRate*2)
   writeShortLE(f,2); writeShortLE(f,16)
   f.writeBytes("data"); writeIntLE(f,dataBytes)
   for(v in samples) writeShortLE(f,(v.coerceIn(-1f,1f)*32767f).roundToInt())
  }
 }

 private fun ascii(b:ByteArray,o:Int,n:Int)=String(b,o,n,StandardCharsets.US_ASCII)
 private fun leShort(b:ByteArray,o:Int)=(b[o].toInt() and 0xff) or ((b[o+1].toInt() and 0xff) shl 8)
 private fun leSignedShort(b:ByteArray,o:Int):Int {
  val v=leShort(b,o); return if(v and 0x8000!=0) v-0x10000 else v
 }
 private fun leSigned24(b:ByteArray,o:Int):Int {
  var v=(b[o].toInt() and 0xff) or ((b[o+1].toInt() and 0xff) shl 8) or ((b[o+2].toInt() and 0xff) shl 16)
  if(v and 0x800000!=0) v=v or -0x1000000
  return v
 }
 private fun leInt(b:ByteArray,o:Int):Int =
  (b[o].toInt() and 0xff) or ((b[o+1].toInt() and 0xff) shl 8) or
   ((b[o+2].toInt() and 0xff) shl 16) or ((b[o+3].toInt() and 0xff) shl 24)
 private fun writeIntLE(f:RandomAccessFile,v:Int) {
  f.write(v and 0xff); f.write((v ushr 8) and 0xff); f.write((v ushr 16) and 0xff); f.write((v ushr 24) and 0xff)
 }
 private fun writeShortLE(f:RandomAccessFile,v:Int) {
  f.write(v and 0xff); f.write((v ushr 8) and 0xff)
 }
}
