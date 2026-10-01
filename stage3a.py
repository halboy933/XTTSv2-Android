from pathlib import Path
import re

root = Path('.')
main_path = root / 'app/src/main/java/com/dorama/xtts/MainActivity.kt'
engine_path = root / 'app/src/main/java/com/dorama/xtts/XttsConditioning.kt'
gradle_path = root / 'app/build.gradle.kts'

if not main_path.exists() or not gradle_path.exists():
    raise SystemExit('Run this script from the XTTSv2-Android repository root.')

main = r'''package com.dorama.xtts

import android.app.Activity
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.util.zip.ZipInputStream

class MainActivity : AppCompatActivity() {
 private lateinit var status: TextView
 private var refPath: String? = null
 private var lastReport: String = ""
 private val modelDir by lazy { File(filesDir, "xtts_models") }

 override fun onCreate(savedInstanceState: Bundle?) {
  super.onCreate(savedInstanceState)

  val content=LinearLayout(this).apply {
   orientation=LinearLayout.VERTICAL
   setPadding(32,32,32,32)
  }
  val scroll=ScrollView(this).apply {
   isFillViewport=true
   addView(content)
  }

  content.addView(TextView(this).apply {
   text="XTTS-v2 Android V2 • Stage 3A"
   textSize=24f
  })
  content.addView(TextView(this).apply {
   text="Reference WAV → mel → conditioning + speaker embedding"
  })

  val import=Button(this).apply { text="Import voice WAV (3–6 sec)" }
  val text=EditText(this).apply {
   hint="Русский текст (будет использоваться на Stage 3B)"
   setText("Сегодня я хочу рассказать вам об одной удивительной истории.")
  }
  val download=Button(this).apply { text="Import model ZIP / verify ONNX" }
  val generate=Button(this).apply { text="Compute voice conditioning (Stage 3A)" }
  val copyReport=Button(this).apply {
   text="Copy Stage 3A report"
   isEnabled=false
  }
  status=TextView(this).apply {
   setTextIsSelectable(true)
  }

  content.addView(import)
  content.addView(text)
  content.addView(download)
  content.addView(generate)
  content.addView(copyReport)
  content.addView(status)
  setContentView(scroll)

  val existingRef=File(filesDir,"reference.wav")
  if(existingRef.exists()) refPath=existingRef.absolutePath

  runCatching { OrtEnvironment.getEnvironment() }
   .onSuccess {
    val modelCount=modelDir.walkTopDown().count { it.isFile && it.extension.equals("onnx",true) }
    status.text=buildString {
     append("Status: ONNX Runtime OK\n")
     append("Reference: ")
     append(if(existingRef.exists()) "found (${existingRef.length()/1024} KB)" else "not imported")
     append("\nONNX models: $modelCount found\n")
     append("Ready for Stage 3A.")
    }
   }
   .onFailure { status.text="ONNX Runtime error: ${it.message}" }

  copyReport.setOnClickListener {
   if(lastReport.isNotBlank()) {
    val clipboard=getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
    clipboard.setPrimaryClip(ClipData.newPlainText("XTTS Stage 3A report",lastReport))
    Toast.makeText(this,"Stage 3A report copied",Toast.LENGTH_SHORT).show()
   }
  }

  import.setOnClickListener {
   startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
    type="audio/*"
    addCategory(Intent.CATEGORY_OPENABLE)
   },7)
  }

  download.setOnClickListener {
   startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
    type="application/zip"
    addCategory(Intent.CATEGORY_OPENABLE)
   },8)
  }

  generate.setOnClickListener {
   val ref=File(filesDir,"reference.wav")
   if(!ref.exists()) {
    status.text="Stage 3A: import reference WAV first."
    return@setOnClickListener
   }

   generate.isEnabled=false
   copyReport.isEnabled=false
   lastReport=""
   status.text="Stage 3A: starting…"

   Thread {
    val result=runCatching {
     XttsConditioning(filesDir).run(ref) { message ->
      runOnUiThread { status.text=message }
     }
    }
    runOnUiThread {
     generate.isEnabled=true
     result.onSuccess {
      lastReport=it
      copyReport.isEnabled=true
      status.text=it
     }.onFailure {
      lastReport=""
      copyReport.isEnabled=false
      status.text="Stage 3A failed: ${it.javaClass.simpleName}: ${it.message}"
     }
    }
   }.start()
  }
 }

 override fun onActivityResult(requestCode:Int,resultCode:Int,data:Intent?) {
  super.onActivityResult(requestCode,resultCode,data)
  if(resultCode!=Activity.RESULT_OK) return
  val u=data?.data?:return

  if(requestCode==7) {
   val f=File(filesDir,"reference.wav")
   contentResolver.openInputStream(u)!!.use { a ->
    f.outputStream().use { b -> a.copyTo(b) }
   }
   refPath=f.absolutePath
   File(filesDir,"conditioning_cache").deleteRecursively()
   lastReport=""
   status.text="Reference imported: ${f.length()/1024} KB\nConditioning cache cleared."
  }

  if(requestCode==8) {
   status.text="Importing model archive…"
   Thread {
    val result=runCatching {
     importModelZip(u)
     validateModels()
    }
    runOnUiThread {
     status.text=result.fold(
      { "ONNX check:\n$it" },
      { "ONNX check failed: ${it.message}" }
     )
    }
   }.start()
  }
 }

 private fun importModelZip(uri:android.net.Uri) {
  modelDir.mkdirs()
  val base=modelDir.canonicalFile
  contentResolver.openInputStream(uri)!!.use { input ->
   ZipInputStream(input).use { zip ->
    var entry=zip.nextEntry
    while(entry!=null) {
     if(!entry.isDirectory) {
      val name=entry.name.replace('\\','/')
      val dest=File(base,name).canonicalFile
      require(dest.path.startsWith(base.path+File.separator)) { "Unsafe archive path" }
      dest.parentFile?.mkdirs()
      dest.outputStream().use { zip.copyTo(it) }
     }
     zip.closeEntry()
     entry=zip.nextEntry
    }
   }
  }
 }

 private fun validateModels():String {
  val models=modelDir.walkTopDown()
   .filter { it.isFile && it.extension.equals("onnx",true) }
   .toList()
  require(models.isNotEmpty()) { "No .onnx files in imported ZIP" }

  val required=setOf(
   "conditioning_encoder.onnx",
   "speaker_encoder.onnx",
   "gpt_model_int8.onnx",
   "hifigan_vocoder.onnx"
  )
  val present=models.map { it.name }.toSet()
  val missing=required-present
  require(missing.isEmpty()) { "Missing models: ${missing.joinToString()}" }

  val env=OrtEnvironment.getEnvironment()
  val opts=OrtSession.SessionOptions()
  try {
   return models.sortedBy { it.name }.joinToString("\n") { model ->
    env.createSession(model.absolutePath,opts).use { session ->
     "${model.name}: ${session.inputNames.size} inputs / ${session.outputNames.size} outputs"
    }
   }
  } finally {
   opts.close()
  }
 }
}
'''

engine = r'''package com.dorama.xtts

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.charset.StandardCharsets
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.log10
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sin

class XttsConditioning(private val filesDir: File) {
 private val modelRoot=File(filesDir,"xtts_models")
 private val env:OrtEnvironment=OrtEnvironment.getEnvironment()

 data class WavData(val samples:FloatArray,val sampleRate:Int)
 data class MelData(val channels:Int,val frames:Int,val data:FloatArray)
 data class TensorResult(val shape:LongArray,val data:FloatArray)

 fun run(referenceFile:File, progress:(String)->Unit):String {
  require(referenceFile.exists()) { "reference.wav not found" }
  val condModel=findRequired("conditioning_encoder.onnx")
  val speakerModel=findRequired("speaker_encoder.onnx")
  val melStatsFile=findRequired("mel_stats.npy")

  val totalStart=System.nanoTime()
  progress("Stage 3A 1/6: reading WAV…")
  val wav=readWav(referenceFile)
  require(wav.samples.isNotEmpty()) { "Reference WAV is empty" }
  val duration=wav.samples.size.toDouble()/wav.sampleRate
  require(duration>=1.0) { "Reference is too short: %.2f s".format(duration) }

  progress("Stage 3A 2/6: resampling reference to 22.05 kHz…")
  val tRes22=System.nanoTime()
  var audio22=resampleSinc(wav.samples,wav.sampleRate,22050)
  val max22=22050*6
  if(audio22.size>max22) audio22=audio22.copyOf(max22)
  val msRes22=elapsedMs(tRes22)

  progress("Stage 3A 3/6: computing 80-bin conditioning mel…")
  val melNorms=readNpyFloat1D(melStatsFile)
  require(melNorms.size==80) { "mel_stats.npy expected 80 values, got ${melNorms.size}" }
  val tMel80=System.nanoTime()
  val mel80=conditioningMel(audio22,melNorms)
  val msMel80=elapsedMs(tMel80)

  progress("Stage 3A 4/6: running conditioning_encoder.onnx…")
  val tCond=System.nanoTime()
  val cond=runModel(
   condModel,
   "mel_spectrogram",
   mel80.data,
   longArrayOf(1,mel80.channels.toLong(),mel80.frames.toLong())
  )
  val msCond=elapsedMs(tCond)
  require(cond.shape.contentEquals(longArrayOf(1,32,1024))) {
   "Unexpected cond_latents shape: ${shape(cond.shape)}"
  }
  require(cond.data.all { it.isFinite() }) { "cond_latents contains NaN/Inf" }

  progress("Stage 3A 5/6: resampling reference to 16 kHz + speaker mel…")
  val tSpkPrep=System.nanoTime()
  val audio16=resampleSinc(wav.samples,wav.sampleRate,16000)
  val mel64=speakerMel(audio16)
  val msSpkPrep=elapsedMs(tSpkPrep)

  progress("Stage 3A 6/6: running speaker_encoder.onnx…")
  val tSpk=System.nanoTime()
  val speaker=runModel(
   speakerModel,
   "mel_spec",
   mel64.data,
   longArrayOf(1,mel64.channels.toLong(),mel64.frames.toLong())
  )
  val msSpk=elapsedMs(tSpk)
  require(speaker.shape.contentEquals(longArrayOf(1,512,1))) {
   "Unexpected speaker_embedding shape: ${shape(speaker.shape)}"
  }
  require(speaker.data.all { it.isFinite() }) { "speaker_embedding contains NaN/Inf" }

  val cacheDir=File(filesDir,"conditioning_cache").apply { mkdirs() }
  writeFloat32LE(File(cacheDir,"cond_latents.f32"),cond.data)
  writeFloat32LE(File(cacheDir,"speaker_embedding.f32"),speaker.data)
  File(cacheDir,"meta.txt").writeText(
   "cond_shape=${shape(cond.shape)}\n"+
    "speaker_shape=${shape(speaker.shape)}\n"+
    "reference_sr=${wav.sampleRate}\n"+
    "reference_samples=${wav.samples.size}\n"
  )

  val totalMs=elapsedMs(totalStart)
  val condStats=stats(cond.data)
  val speakerStats=stats(speaker.data)

  return buildString {
   append("STAGE 3A SUCCESS\n\n")
   append("Reference: ${wav.sampleRate} Hz, ${"%.3f".format(duration)} s, ${wav.samples.size} samples\n")
   append("22.05 kHz samples: ${audio22.size} (${msRes22} ms)\n")
   append("conditioning mel: [1, ${mel80.channels}, ${mel80.frames}] (${msMel80} ms)\n")
   append("cond_latents: ${shape(cond.shape)} (${msCond} ms)\n")
   append("  meanAbs=${"%.6f".format(condStats.first)} maxAbs=${"%.6f".format(condStats.second)}\n\n")
   append("16 kHz samples: ${audio16.size}\n")
   append("speaker mel: [1, ${mel64.channels}, ${mel64.frames}] (${msSpkPrep} ms incl. resample)\n")
   append("speaker_embedding: ${shape(speaker.shape)} (${msSpk} ms)\n")
   append("  meanAbs=${"%.6f".format(speakerStats.first)} maxAbs=${"%.6f".format(speakerStats.second)}\n\n")
   append("Cache saved: conditioning_cache/\n")
   append("Total: ${totalMs} ms\n")
   append("Ready for Stage 3B (Russian tokenizer + GPT INT8).")
  }
 }

 private fun findRequired(name:String):File {
  val f=modelRoot.walkTopDown().firstOrNull { it.isFile && it.name==name }
  require(f!=null) { "$name not found. Import the model ZIP first." }
  return f
 }

 private fun runModel(model:File,inputName:String,data:FloatArray,inputShape:LongArray):TensorResult {
  val opts=OrtSession.SessionOptions()
  try {
   env.createSession(model.absolutePath,opts).use { session ->
    OnnxTensor.createTensor(env,FloatBuffer.wrap(data),inputShape).use { tensor ->
     session.run(mapOf(inputName to tensor)).use { result ->
      require(result.size()>0) { "${model.name} returned no outputs" }
      val value=result[0]
      val info=value.info as? TensorInfo ?: error("${model.name} output is not a tensor")
      return TensorResult(info.shape,flattenFloatTensor(value.value))
     }
    }
   }
  } finally {
   opts.close()
  }
 }

 private fun flattenFloatTensor(value:Any?):FloatArray {
  val out=ArrayList<Float>()
  fun visit(v:Any?) {
   when(v) {
    null -> Unit
    is Float -> out.add(v)
    is Double -> out.add(v.toFloat())
    is FloatArray -> for(x in v) out.add(x)
    is DoubleArray -> for(x in v) out.add(x.toFloat())
    is Array<*> -> for(x in v) visit(x)
    else -> if(v.javaClass.isArray) {
     val n=java.lang.reflect.Array.getLength(v)
     for(i in 0 until n) visit(java.lang.reflect.Array.get(v,i))
    } else error("Unsupported ONNX tensor value: ${v.javaClass.name}")
   }
  }
  visit(value)
  return FloatArray(out.size) { out[it] }
 }

 private fun stats(a:FloatArray):Pair<Double,Double> {
  var sum=0.0
  var maxAbs=0.0
  for(v in a) {
   val x=abs(v.toDouble())
   sum+=x
   if(x>maxAbs) maxAbs=x
  }
  return Pair(if(a.isEmpty()) 0.0 else sum/a.size,maxAbs)
 }

 private fun readWav(file:File):WavData {
  val b=file.readBytes()
  require(b.size>=44) { "WAV file too small" }
  require(ascii(b,0,4)=="RIFF" && ascii(b,8,4)=="WAVE") { "Only RIFF/WAVE is supported" }

  var pos=12
  var format=-1
  var channels=-1
  var sampleRate=-1
  var bits=-1
  var blockAlign=-1
  var dataOffset=-1
  var dataSize=-1

  while(pos+8<=b.size) {
   val id=ascii(b,pos,4)
   val size=leInt(b,pos+4)
   val body=pos+8
   require(size>=0 && body+size<=b.size) { "Invalid WAV chunk $id" }
   when(id) {
    "fmt " -> {
     require(size>=16) { "Invalid WAV fmt chunk" }
     format=leShort(b,body)
     channels=leShort(b,body+2)
     sampleRate=leInt(b,body+4)
     blockAlign=leShort(b,body+12)
     bits=leShort(b,body+14)
    }
    "data" -> {
     dataOffset=body
     dataSize=size
    }
   }
   pos=body+size+(size and 1)
  }

  require(dataOffset>=0 && dataSize>0) { "WAV data chunk not found" }
  require(channels>0 && sampleRate>0 && blockAlign>0) { "Invalid WAV format" }
  require(format==1 || format==3) { "WAV format $format unsupported (PCM/float only)" }

  val frames=dataSize/blockAlign
  val mono=FloatArray(frames)
  val bytesPerSample=(bits+7)/8
  require(bytesPerSample*channels<=blockAlign) { "Invalid WAV blockAlign" }

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

 private fun resampleSinc(input:FloatArray,srcRate:Int,dstRate:Int):FloatArray {
  if(srcRate==dstRate) return input.copyOf()
  val ratio=dstRate.toDouble()/srcRate
  val outLen=max(1,(input.size*ratio).roundToInt())
  val out=FloatArray(outLen)
  val radius=32
  val cutoff=min(1.0,ratio)*0.97

  for(i in 0 until outLen) {
   val srcPos=i/ratio
   val center=floor(srcPos).toInt()
   var acc=0.0
   var wsum=0.0
   val start=center-radius+1
   val end=center+radius
   for(k in start..end) {
    if(k<0 || k>=input.size) continue
    val x=srcPos-k
    val ax=abs(x)
    if(ax>=radius) continue
    val pix=PI*cutoff*x
    val sinc=if(abs(pix)<1e-12) 1.0 else sin(pix)/pix
    val window=0.5+0.5*cos(PI*x/radius)
    val w=sinc*window
    acc+=input[k]*w
    wsum+=w
   }
   out[i]=if(abs(wsum)>1e-12) (acc/wsum).toFloat() else 0f
  }
  return out
 }

 private fun conditioningMel(audio:FloatArray,melNorms:FloatArray):MelData {
  val nFft=2048
  val hop=256
  val winLength=1024
  val nMels=80
  val sr=22050
  val pad=nFft/2
  val x=reflectPad(audio,pad)
  val frames=1+(x.size-nFft)/hop
  require(frames>0) { "Reference too short for conditioning STFT" }

  val window=FloatArray(nFft)
  val offset=(nFft-winLength)/2
  for(n in 0 until winLength) {
   window[offset+n]=(0.5-0.5*cos(2.0*PI*n/(winLength-1))).toFloat()
  }
  val fb=melFilterBank(nFft,sr,0.0,8000.0,nMels,slaneyNorm=true)
  val fft=Radix2Fft(nFft)
  val frame=FloatArray(nFft)
  val out=FloatArray(nMels*frames)

  for(f in 0 until frames) {
   val base=f*hop
   for(i in 0 until nFft) frame[i]=x[base+i]*window[i]
   val power=fft.power(frame)
   for(m in 0 until nMels) {
    var sum=0.0
    val filt=fb[m]
    for(k in power.indices) sum+=power[k]*filt[k]
    val logMel=ln(max(sum,1e-5))
    out[m*frames+f]=(logMel/melNorms[m]).toFloat()
   }
  }
  return MelData(nMels,frames,out)
 }

 private fun speakerMel(input:FloatArray):MelData {
  val nFft=512
  val hop=160
  val winLength=400
  val nMels=64
  val sr=16000
  val pre=FloatArray(input.size)
  if(input.size==1) {
   pre[0]=input[0]*(1f-0.97f)
  } else {
   val padLeft=2f*input[0]-input[1]
   pre[0]=input[0]-0.97f*padLeft
   for(i in 1 until input.size) pre[i]=input[i]-0.97f*input[i-1]
  }

  val x=reflectPad(pre,nFft/2)
  val frames=1+(x.size-nFft)/hop
  require(frames>0) { "Reference too short for speaker STFT" }

  val window=FloatArray(nFft)
  val offset=(nFft-winLength)/2
  for(n in 0 until winLength) {
   window[offset+n]=(0.54-0.46*cos(2.0*PI*n/(winLength-1))).toFloat()
  }
  val fb=melFilterBank(nFft,sr,0.0,(sr/2).toDouble(),nMels,slaneyNorm=false)
  val fft=Radix2Fft(nFft)
  val frame=FloatArray(nFft)
  val out=FloatArray(nMels*frames)

  for(f in 0 until frames) {
   val base=f*hop
   for(i in 0 until nFft) frame[i]=x[base+i]*window[i]
   val power=fft.power(frame)
   for(m in 0 until nMels) {
    var sum=0.0
    val filt=fb[m]
    for(k in power.indices) sum+=power[k]*filt[k]
    out[m*frames+f]=sum.toFloat()
   }
  }
  return MelData(nMels,frames,out)
 }

 private fun reflectPad(x:FloatArray,pad:Int):FloatArray {
  require(x.size>pad) { "Reference too short for reflect padding" }
  val out=FloatArray(x.size+2*pad)
  for(i in 0 until pad) out[i]=x[pad-i]
  System.arraycopy(x,0,out,pad,x.size)
  for(i in 0 until pad) out[pad+x.size+i]=x[x.size-2-i]
  return out
 }

 private fun melFilterBank(
  nFft:Int,
  sampleRate:Int,
  fMin:Double,
  fMax:Double,
  nMels:Int,
  slaneyNorm:Boolean
 ):Array<FloatArray> {
  val nFreq=nFft/2+1
  val mMin=hzToMelHtk(fMin)
  val mMax=hzToMelHtk(fMax)
  val hzPts=DoubleArray(nMels+2) { i ->
   val mel=mMin+(mMax-mMin)*i/(nMels+1)
   melToHzHtk(mel)
  }
  val filters=Array(nMels) { FloatArray(nFreq) }
  for(m in 0 until nMels) {
   val left=hzPts[m]
   val center=hzPts[m+1]
   val right=hzPts[m+2]
   val norm=if(slaneyNorm) 2.0/(right-left) else 1.0
   for(k in 0 until nFreq) {
    val freq=sampleRate.toDouble()*k/nFft
    val v=when {
     freq<left || freq>right -> 0.0
     freq<=center -> (freq-left)/(center-left+1e-10)
     else -> (right-freq)/(right-center+1e-10)
    }
    filters[m][k]=(max(0.0,v)*norm).toFloat()
   }
  }
  return filters
 }

 private fun hzToMelHtk(hz:Double)=2595.0*log10(1.0+hz/700.0)
 private fun melToHzHtk(mel:Double)=700.0*(10.0.pow(mel/2595.0)-1.0)

 private class Radix2Fft(private val n:Int) {
  private val cosTable=DoubleArray(n/2) { k -> cos(2.0*PI*k/n) }
  private val sinTable=DoubleArray(n/2) { k -> -sin(2.0*PI*k/n) }
  private val re=DoubleArray(n)
  private val im=DoubleArray(n)

  init { require(n>0 && n and (n-1)==0) { "FFT size must be power of two" } }

  fun power(input:FloatArray):DoubleArray {
   require(input.size==n)
   for(i in 0 until n) {
    re[i]=input[i].toDouble()
    im[i]=0.0
   }

   var j=0
   for(i in 1 until n) {
    var bit=n shr 1
    while(j and bit !=0) {
     j=j xor bit
     bit=bit shr 1
    }
    j=j xor bit
    if(i<j) {
     val tr=re[i]; re[i]=re[j]; re[j]=tr
     val ti=im[i]; im[i]=im[j]; im[j]=ti
    }
   }

   var len=2
   while(len<=n) {
    val half=len/2
    val tableStep=n/len
    var base=0
    while(base<n) {
     for(k in 0 until half) {
      val idx=k*tableStep
      val c=cosTable[idx]
      val s=sinTable[idx]
      val r2=re[base+k+half]
      val i2=im[base+k+half]
      val tr=r2*c-i2*s
      val ti=r2*s+i2*c
      val p=base+k
      val q=p+half
      re[q]=re[p]-tr
      im[q]=im[p]-ti
      re[p]+=tr
      im[p]+=ti
     }
     base+=len
    }
    len=len shl 1
   }

   return DoubleArray(n/2+1) { k -> re[k]*re[k]+im[k]*im[k] }
  }
 }

 private fun readNpyFloat1D(file:File):FloatArray {
  val b=file.readBytes()
  require(b.size>=12) { "Invalid NPY file" }
  require((b[0].toInt() and 0xff)==0x93 && ascii(b,1,5)=="NUMPY") { "Invalid NPY magic" }
  val major=b[6].toInt() and 0xff
  val headerLen:Int
  val dataOffset:Int
  if(major==1) {
   headerLen=(b[8].toInt() and 0xff) or ((b[9].toInt() and 0xff) shl 8)
   dataOffset=10+headerLen
  } else {
   headerLen=leInt(b,8)
   dataOffset=12+headerLen
  }
  require(dataOffset in 1..b.size) { "Invalid NPY header" }
  val header=String(b,if(major==1) 10 else 12,headerLen,StandardCharsets.US_ASCII)
  require(header.contains("f4")) { "Expected float32 NPY, header=$header" }
  require((b.size-dataOffset)%4==0) { "Invalid float32 NPY size" }
  val n=(b.size-dataOffset)/4
  val out=FloatArray(n)
  val bb=ByteBuffer.wrap(b,dataOffset,b.size-dataOffset).order(ByteOrder.LITTLE_ENDIAN)
  for(i in 0 until n) out[i]=bb.float
  return out
 }

 private fun writeFloat32LE(file:File,data:FloatArray) {
  val bb=ByteBuffer.allocate(data.size*4).order(ByteOrder.LITTLE_ENDIAN)
  for(v in data) bb.putFloat(v)
  file.writeBytes(bb.array())
 }

 private fun ascii(b:ByteArray,o:Int,n:Int)=String(b,o,n,StandardCharsets.US_ASCII)
 private fun leShort(b:ByteArray,o:Int)=(b[o].toInt() and 0xff) or ((b[o+1].toInt() and 0xff) shl 8)
 private fun leSignedShort(b:ByteArray,o:Int):Int {
  val v=leShort(b,o)
  return if(v and 0x8000 !=0) v-0x10000 else v
 }
 private fun leSigned24(b:ByteArray,o:Int):Int {
  var v=(b[o].toInt() and 0xff) or ((b[o+1].toInt() and 0xff) shl 8) or ((b[o+2].toInt() and 0xff) shl 16)
  if(v and 0x800000 !=0) v=v or -0x1000000
  return v
 }
 private fun leInt(b:ByteArray,o:Int):Int =
  (b[o].toInt() and 0xff) or
   ((b[o+1].toInt() and 0xff) shl 8) or
   ((b[o+2].toInt() and 0xff) shl 16) or
   ((b[o+3].toInt() and 0xff) shl 24)

 private fun elapsedMs(start:Long)=((System.nanoTime()-start)/1_000_000L)
 private fun shape(s:LongArray)="[${s.joinToString(", ")}]"
}
'''

main_path.write_text(main, encoding='utf-8')
engine_path.write_text(engine, encoding='utf-8')

gradle = gradle_path.read_text(encoding='utf-8')
gradle = re.sub(r'versionCode\s*=\s*\d+', 'versionCode = 4', gradle, count=1)
gradle = re.sub(r'versionName\s*=\s*"[^"]+"', 'versionName = "0.3.0-stage3a"', gradle, count=1)
if 'XTTS_KEYSTORE_PATH' not in gradle or 'signingConfigs' not in gradle:
    raise SystemExit('Persistent signing block is missing; refusing to overwrite Gradle config.')
gradle_path.write_text(gradle, encoding='utf-8')

print('Stage 3A applied successfully')
print('Added: XttsConditioning.kt')
print('MainActivity: real reference conditioning test')
print('Version: code 4 / 0.3.0-stage3a')
print('Persistent signing block preserved')
