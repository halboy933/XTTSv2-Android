package com.dorama.xtts

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import java.util.Locale
import kotlin.math.min
import kotlin.math.roundToInt

class XttsLongSynthesis(private val filesDir:File) {
 data class Chunk(
  val index:Int,
  val text:String,
  val bpeTokens:Int,
  val outputFile:File
 )

 data class Result(
  val outputFile:File,
  val chunks:List<Chunk>,
  val similarity:Double,
  val durationSec:Double,
  val report:String
 )

 private data class PcmWav(
  val sampleRate:Int,
  val channels:Int,
  val bits:Int,
  val samples:ShortArray
 )

 private val modelRoot=File(filesDir,"xtts_models")

 fun run(
  text:String,
  preset:Int,
  seed:Int,
  progress:(String)->Unit
 ):Result {
  require(text.isNotBlank()) { "Enter text first." }
  val tokenizer=XttsBpeTokenizer(findRequired("vocab.json"))

  progress("Stage 3L 1/4: делю длинный текст на безопасные фрагменты…")
  val texts=splitText(text,tokenizer,maxBpe=90)
  require(texts.isNotEmpty()) { "Text produced no chunks." }
  require(texts.size<=80) { "Too many chunks: ${texts.size}. Split the text into smaller jobs." }

  val chunks=ArrayList<Chunk>()
  val reports=ArrayList<String>()
  val sampling=when(preset) {
   1 -> XttsSynthesisStage3C.Sampling(
    temperature=0.65f,
    topK=30,
    topP=0.90f,
    repetitionPenalty=10.0f,
    seed=seed
   )
   else -> XttsSynthesisStage3C.Sampling(seed=seed)
  }

  val partFiles=ArrayList<File>()
  try {
   for(i in texts.indices) {
    val chunkText=texts[i]
    val bpe=tokenizer.encodeRussian(chunkText).ids.size
    val name="xtts_long_part_${String.format(Locale.US,"%03d",i+1)}.wav"
    val part=File(filesDir,name)
    part.delete()

    progress(
     "Stage 3L 2/4: фрагмент ${i+1}/${texts.size}\n"+
      "$bpe BPE токенов\n$chunkText"
    )

    val report=XttsSynthesisStage3C(filesDir).run(
     chunkText,
     sampling,
     name
    ) { message ->
     progress("Stage 3L 2/4: фрагмент ${i+1}/${texts.size}\n$message")
    }

    if(report.contains("Stop token reached: false")) {
     throw IllegalStateException(
      "Фрагмент ${i+1} не завершился STOP-токеном. "+
       "Текст этого фрагмента нужно сократить ещё сильнее."
     )
    }

    partFiles.add(part)
    chunks.add(Chunk(i+1,chunkText,bpe,part))
    reports.add(
     "CHUNK ${i+1}/${texts.size} • $bpe BPE\n"+
      chunkText+"\n"+
      report
    )
   }

   progress("Stage 3L 3/4: склеиваю ${partFiles.size} WAV-фрагментов…")
   val output=File(filesDir,"xtts_long.wav")
   mergePcmWavs(
    partFiles,
    output,
    pauseMs=180,
    fadeMs=6
   )

   val wav=readPcm16(output)
   val duration=wav.samples.size.toDouble()/wav.sampleRate

   progress("Stage 3L 4/4: проверяю сходство итогового голоса…")
   val similarity=runCatching {
    XttsVoiceSimilarity(filesDir).score(output)
   }.getOrElse { Double.NaN }

   val report=buildString {
    append("STAGE 3L SUCCESS — long text → one WAV\n\n")
    append("Chunks: ${chunks.size}\n")
    append("Safe limit: 90 BPE tokens per chunk\n")
    append("Seed: $seed\n")
    append("Pause between chunks: 180 ms\n")
    append("Edge fade: 6 ms\n")
    append("Output: ${output.name}\n")
    append("Duration: ${String.format(Locale.US,"%.2f",duration)} s\n")
    if(similarity.isFinite()) {
     append("Voice similarity: ${String.format(Locale.US,"%.4f",similarity)}\n")
    }
    append("\nChunk map:\n")
    for(chunk in chunks) {
     append("${chunk.index}. ${chunk.bpeTokens} BPE • ${chunk.text}\n")
    }
    append("\nEach chunk is generated independently, so the GPT audio-token limit resets at every boundary.")
   }

   return Result(
    outputFile=output,
    chunks=chunks,
    similarity=similarity,
    durationSec=duration,
    report=report
   )
  } finally {
   for(file in partFiles) runCatching { file.delete() }
  }
 }

 private fun splitText(
  source:String,
  tokenizer:XttsBpeTokenizer,
  maxBpe:Int
 ):List<String> {
  val normalized=source
   .replace(Regex("[\\t ]+")," ")
   .replace(Regex("\\n{2,}"),"\n")
   .trim()
  require(normalized.isNotBlank()) { "Text is empty after normalization." }

  val sentenceUnits=normalized
   .split(Regex("(?<=[.!?…])\\s+|\\n+"))
   .map { it.trim() }
   .filter { it.isNotBlank() }

  val atomic=ArrayList<String>()
  for(unit in sentenceUnits) {
   if(tokenCount(tokenizer,unit)<=maxBpe) {
    atomic.add(unit)
   } else {
    atomic.addAll(splitOversizeUnit(unit,tokenizer,maxBpe))
   }
  }

  val result=ArrayList<String>()
  var current=""
  for(unit in atomic) {
   val candidate=if(current.isBlank()) unit else "$current $unit"
   if(tokenCount(tokenizer,candidate)<=maxBpe) {
    current=candidate
   } else {
    if(current.isNotBlank()) result.add(current.trim())
    current=unit
   }
  }
  if(current.isNotBlank()) result.add(current.trim())

  for((i,chunk) in result.withIndex()) {
   val count=tokenCount(tokenizer,chunk)
   require(count in 1..maxBpe) {
    "Chunk ${i+1} has $count BPE tokens, expected <= $maxBpe"
   }
  }
  return result
 }

 private fun splitOversizeUnit(
  unit:String,
  tokenizer:XttsBpeTokenizer,
  maxBpe:Int
 ):List<String> {
  val words=unit.split(Regex("\\s+")).filter { it.isNotBlank() }
  require(words.isNotEmpty()) { "Cannot split long sentence." }

  val out=ArrayList<String>()
  var current=""

  for(word in words) {
   val candidate=if(current.isBlank()) word else "$current $word"
   val count=tokenCount(tokenizer,candidate)
   if(count<=maxBpe) {
    current=candidate
   } else {
    if(current.isNotBlank()) out.add(current.trim())
    require(tokenCount(tokenizer,word)<=maxBpe) {
     "One token group is too large to split safely: $word"
    }
    current=word
   }
  }
  if(current.isNotBlank()) out.add(current.trim())
  return out
 }

 private fun tokenCount(tokenizer:XttsBpeTokenizer,text:String):Int =
  tokenizer.encodeRussian(text).ids.size

 private fun mergePcmWavs(
  files:List<File>,
  output:File,
  pauseMs:Int,
  fadeMs:Int
 ) {
  require(files.isNotEmpty()) { "No WAV chunks to merge." }
  val wavs=files.map { readPcm16(it) }
  val first=wavs.first()
  require(first.channels==1 && first.bits==16) {
   "Expected mono PCM16 WAV, got channels=${first.channels}, bits=${first.bits}"
  }
  for((i,wav) in wavs.withIndex()) {
   require(wav.sampleRate==first.sampleRate) {
    "Chunk ${i+1} sample rate ${wav.sampleRate}, expected ${first.sampleRate}"
   }
   require(wav.channels==1 && wav.bits==16) {
    "Chunk ${i+1} must be mono PCM16"
   }
  }

  val pauseSamples=(first.sampleRate*pauseMs/1000.0).roundToInt()
  val fadeSamples=(first.sampleRate*fadeMs/1000.0).roundToInt().coerceAtLeast(1)
  val totalSamples=wavs.sumOf { it.samples.size }+
   pauseSamples*(wavs.size-1).coerceAtLeast(0)

  val merged=ShortArray(totalSamples)
  var pos=0
  for((index,wav) in wavs.withIndex()) {
   val samples=wav.samples.copyOf()
   applyEdgeFade(samples,fadeSamples)
   System.arraycopy(samples,0,merged,pos,samples.size)
   pos+=samples.size
   if(index<wavs.lastIndex) pos+=pauseSamples
  }
  writePcm16(output,merged,first.sampleRate)
 }

 private fun applyEdgeFade(samples:ShortArray,fadeSamples:Int) {
  val n=min(fadeSamples,samples.size/2)
  if(n<=0) return
  for(i in 0 until n) {
   val gain=(i+1).toDouble()/n
   samples[i]=(samples[i]*gain).roundToInt()
    .coerceIn(Short.MIN_VALUE.toInt(),Short.MAX_VALUE.toInt())
    .toShort()
   val j=samples.lastIndex-i
   samples[j]=(samples[j]*gain).roundToInt()
    .coerceIn(Short.MIN_VALUE.toInt(),Short.MAX_VALUE.toInt())
    .toShort()
  }
 }

 private fun readPcm16(file:File):PcmWav {
  val b=file.readBytes()
  require(b.size>=44) { "WAV too small: ${file.name}" }
  require(ascii(b,0,4)=="RIFF" && ascii(b,8,4)=="WAVE") {
   "Invalid WAV: ${file.name}"
  }

  var pos=12
  var format=-1
  var channels=-1
  var sampleRate=-1
  var bits=-1
  var dataOffset=-1
  var dataSize=-1

  while(pos+8<=b.size) {
   val id=ascii(b,pos,4)
   val size=leInt(b,pos+4)
   val body=pos+8
   require(size>=0 && body+size<=b.size) { "Invalid WAV chunk in ${file.name}" }
   when(id) {
    "fmt " -> {
     require(size>=16) { "Invalid fmt chunk" }
     format=leShort(b,body)
     channels=leShort(b,body+2)
     sampleRate=leInt(b,body+4)
     bits=leShort(b,body+14)
    }
    "data" -> {
     dataOffset=body
     dataSize=size
    }
   }
   pos=body+size+(size and 1)
  }

  require(format==1) { "Only PCM WAV supported for merge: format=$format" }
  require(channels==1 && bits==16) {
   "Expected mono PCM16, got channels=$channels bits=$bits"
  }
  require(sampleRate>0 && dataOffset>=0 && dataSize>0 && dataSize%2==0) {
   "Invalid WAV data in ${file.name}"
  }

  val bb=ByteBuffer.wrap(b,dataOffset,dataSize).order(ByteOrder.LITTLE_ENDIAN)
  val samples=ShortArray(dataSize/2) { bb.short }
  return PcmWav(sampleRate,channels,bits,samples)
 }

 private fun writePcm16(file:File,samples:ShortArray,sampleRate:Int) {
  val dataBytes=samples.size*2
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
   for(sample in samples) writeShortLE(f,sample.toInt())
  }
 }

 private fun findRequired(name:String):File {
  val f=modelRoot.walkTopDown().firstOrNull { it.isFile && it.name==name }
  require(f!=null) { "$name not found. Import model ZIP first." }
  return f
 }

 private fun ascii(b:ByteArray,o:Int,n:Int)=String(b,o,n,StandardCharsets.US_ASCII)
 private fun leShort(b:ByteArray,o:Int) =
  (b[o].toInt() and 0xff) or ((b[o+1].toInt() and 0xff) shl 8)

 private fun leInt(b:ByteArray,o:Int):Int =
  (b[o].toInt() and 0xff) or
   ((b[o+1].toInt() and 0xff) shl 8) or
   ((b[o+2].toInt() and 0xff) shl 16) or
   ((b[o+3].toInt() and 0xff) shl 24)

 private fun writeIntLE(f:RandomAccessFile,v:Int) {
  f.write(v and 0xff)
  f.write((v ushr 8) and 0xff)
  f.write((v ushr 16) and 0xff)
  f.write((v ushr 24) and 0xff)
 }

 private fun writeShortLE(f:RandomAccessFile,v:Int) {
  f.write(v and 0xff)
  f.write((v ushr 8) and 0xff)
 }
}
