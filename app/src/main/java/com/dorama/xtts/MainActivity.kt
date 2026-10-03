package com.dorama.xtts

import android.app.Activity
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.media.MediaPlayer
import android.os.Bundle
import android.text.InputType
import android.view.Gravity
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.util.zip.ZipInputStream

class MainActivity : AppCompatActivity() {
 private lateinit var status: TextView
 private var refPath: String? = null
 private var lastReport: String = ""
 private var player: MediaPlayer? = null
 private var pendingSaveName: String? = null
 private val generatedNames=ArrayList<String>()
 private lateinit var resultMode: Spinner
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
   text="XTTS-v2 Android V2 • Stage 3F"
   textSize=24f
  })
  content.addView(TextView(this).apply {
   text="Russian XTTS-v2 • reproducible seeds + 3 variants"
  })

  val import=Button(this).apply { text="Import voice WAV (3–6 sec)" }
  val inputText=EditText(this).apply {
   hint="Русский текст"
   setText("Сегодня я хочу рассказать вам об одной удивительной истории.")
   minLines=4
   gravity=Gravity.TOP or Gravity.START
   inputType=InputType.TYPE_CLASS_TEXT or
    InputType.TYPE_TEXT_FLAG_MULTI_LINE or
    InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
   setHorizontallyScrolling(false)
  }
  val qualityLabel=TextView(this).apply {
   text="Режим качества"
  }
  val qualityMode=Spinner(this).apply {
   adapter=ArrayAdapter(
    this@MainActivity,
    android.R.layout.simple_spinner_dropdown_item,
    listOf(
     "Оригинал 3C • 0.75 / 50 / 0.85",
     "Чётче 3E • 0.65 / 30 / 0.90"
    )
   )
   setSelection(0)
  }
  val seedLabel=TextView(this).apply {
   text="Seed (одинаковый seed = воспроизводимая генерация)"
  }
  val seedInput=EditText(this).apply {
   hint="Seed"
   setText("12345")
   inputType=InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_SIGNED
  }
  val resultLabel=TextView(this).apply {
   text="Результат для Play / Save / Share"
  }
  resultMode=Spinner(this)
  val existingGenerated=filesDir.listFiles()
   ?.filter { it.isFile && it.extension.equals("wav",true) && it.name.startsWith("xtts_") }
   ?.sortedByDescending { it.lastModified() }
   ?.map { it.name }
   .orEmpty()
  generatedNames.addAll(existingGenerated)
  if(generatedNames.isEmpty()) generatedNames.add("xtts_generated.wav")
  resultMode.adapter=ArrayAdapter(
   this,
   android.R.layout.simple_spinner_dropdown_item,
   generatedNames
  )
  val download=Button(this).apply { text="Import model ZIP / verify ONNX" }
  val conditioning=Button(this).apply { text="Compute voice conditioning (Stage 3A)" }
  val gptTest=Button(this).apply { text="Test Russian GPT (Stage 3B)" }
  val synthesize=Button(this).apply { text="Generate 1 variant (Stage 3F)" }
  val synthesize3=Button(this).apply { text="Generate 3 variants (seed, seed+1, seed+2)" }
  val play=Button(this).apply {
   text="Play generated WAV"
   isEnabled=File(filesDir,"xtts_generated.wav").exists()
  }
  val saveWav=Button(this).apply {
   text="Save WAV"
   isEnabled=File(filesDir,"xtts_generated.wav").exists()
  }
  val shareWav=Button(this).apply {
   text="Share WAV"
   isEnabled=File(filesDir,"xtts_generated.wav").exists()
  }
  val copyReport=Button(this).apply {
   text="Copy report"
   isEnabled=false
  }
  status=TextView(this).apply { setTextIsSelectable(true) }

  content.addView(import)
  content.addView(inputText)
  content.addView(qualityLabel)
  content.addView(qualityMode)
  content.addView(seedLabel)
  content.addView(seedInput)
  content.addView(resultLabel)
  content.addView(resultMode)
  content.addView(download)
  content.addView(conditioning)
  content.addView(gptTest)
  content.addView(synthesize)
  content.addView(synthesize3)
  content.addView(play)
  content.addView(saveWav)
  content.addView(shareWav)
  content.addView(copyReport)
  content.addView(status)
  setContentView(scroll)

  val existingRef=File(filesDir,"reference.wav")
  if(existingRef.exists()) refPath=existingRef.absolutePath

  runCatching { OrtEnvironment.getEnvironment() }
   .onSuccess {
    val modelCount=modelDir.walkTopDown().count { it.isFile && it.extension.equals("onnx",true) }
    val cache=File(filesDir,"conditioning_cache/cond_latents.f32")
    status.text=buildString {
     append("Status: ONNX Runtime OK\n")
     append("Reference: ")
     append(if(existingRef.exists()) "found (${existingRef.length()/1024} KB)" else "not imported")
     append("\nONNX models: $modelCount found\n")
     append("Conditioning cache: ${if(cache.exists()) "found" else "missing"}\n")
     append("Generated WAV: ${if(File(filesDir,"xtts_generated.wav").exists()) "found" else "missing"}\n")
     append("Ready for Stage 3A / 3B / 3C / 3D / 3E / 3F.")
    }
   }
   .onFailure { status.text="ONNX Runtime error: ${it.message}" }

  copyReport.setOnClickListener {
   if(lastReport.isNotBlank()) {
    val clipboard=getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
    clipboard.setPrimaryClip(ClipData.newPlainText("XTTS report",lastReport))
    Toast.makeText(this,"Report copied",Toast.LENGTH_SHORT).show()
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

  conditioning.setOnClickListener {
   val ref=File(filesDir,"reference.wav")
   if(!ref.exists()) {
    status.text="Stage 3A: import reference WAV first."
    return@setOnClickListener
   }
   conditioning.isEnabled=false
   gptTest.isEnabled=false
   synthesize.isEnabled=false
   synthesize3.isEnabled=false
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
     conditioning.isEnabled=true
     gptTest.isEnabled=true
     synthesize.isEnabled=true
     synthesize3.isEnabled=true
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

  gptTest.setOnClickListener {
   val typed=inputText.text?.toString().orEmpty()
   if(typed.isBlank()) {
    status.text="Stage 3B: enter Russian text first."
    return@setOnClickListener
   }
   conditioning.isEnabled=false
   gptTest.isEnabled=false
   synthesize.isEnabled=false
   synthesize3.isEnabled=false
   copyReport.isEnabled=false
   lastReport=""
   status.text="Stage 3B: starting…"

   Thread {
    val result=runCatching {
     XttsGptStage3B(filesDir).run(typed) { message ->
      runOnUiThread { status.text=message }
     }
    }
    runOnUiThread {
     conditioning.isEnabled=true
     gptTest.isEnabled=true
     synthesize.isEnabled=true
     synthesize3.isEnabled=true
     result.onSuccess {
      lastReport=it
      copyReport.isEnabled=true
      status.text=it
     }.onFailure {
      lastReport=""
      copyReport.isEnabled=false
      status.text="Stage 3B failed: ${it.javaClass.simpleName}: ${it.message}"
     }
    }
   }.start()
  }

  fun selectedSampling(seed:Int):XttsSynthesisStage3C.Sampling {
   return when(qualityMode.selectedItemPosition) {
    1 -> XttsSynthesisStage3C.Sampling(
     temperature=0.65f,
     topK=30,
     topP=0.90f,
     repetitionPenalty=10.0f,
     seed=seed
    )
    else -> XttsSynthesisStage3C.Sampling(seed=seed)
   }
  }

  fun refreshResults(names:List<String>,selectName:String) {
   for(name in names.reversed()) {
    generatedNames.remove(name)
    generatedNames.add(0,name)
   }
   resultMode.adapter=ArrayAdapter(
    this,
    android.R.layout.simple_spinner_dropdown_item,
    generatedNames
   )
   resultMode.setSelection(generatedNames.indexOf(selectName).coerceAtLeast(0))
  }

  fun setGenerationBusy(busy:Boolean) {
   conditioning.isEnabled=!busy
   gptTest.isEnabled=!busy
   synthesize.isEnabled=!busy
   synthesize3.isEnabled=!busy
   qualityMode.isEnabled=!busy
   seedInput.isEnabled=!busy
   resultMode.isEnabled=!busy
   if(busy) {
    play.isEnabled=false
    saveWav.isEnabled=false
    shareWav.isEnabled=false
    copyReport.isEnabled=false
   }
  }

  synthesize.setOnClickListener {
   val typed=inputText.text?.toString().orEmpty()
   if(typed.isBlank()) {
    status.text="Stage 3F: enter Russian text first."
    return@setOnClickListener
   }
   val seed=seedInput.text?.toString()?.trim()?.toIntOrNull()
   if(seed==null) {
    status.text="Stage 3F: seed must be a whole number."
    return@setOnClickListener
   }
   val sampling=selectedSampling(seed)
   val outputName="xtts_seed_${seed}.wav"

   setGenerationBusy(true)
   lastReport=""
   status.text="Stage 3F: generating seed $seed…"

   Thread {
    val result=runCatching {
     XttsSynthesisStage3C(filesDir).run(
      typed,
      sampling,
      outputName
     ) { message ->
      runOnUiThread { status.text="Seed $seed\\n$message" }
     }
    }
    runOnUiThread {
     setGenerationBusy(false)
     result.onSuccess {
      lastReport=it
      refreshResults(listOf(outputName),outputName)
      copyReport.isEnabled=true
      play.isEnabled=true
      saveWav.isEnabled=true
      shareWav.isEnabled=true
      status.text=it
     }.onFailure {
      lastReport=""
      status.text="Stage 3F failed: ${it.javaClass.simpleName}: ${it.message}"
     }
    }
   }.start()
  }

  synthesize3.setOnClickListener {
   val typed=inputText.text?.toString().orEmpty()
   if(typed.isBlank()) {
    status.text="Stage 3F: enter Russian text first."
    return@setOnClickListener
   }
   val baseSeed=seedInput.text?.toString()?.trim()?.toIntOrNull()
   if(baseSeed==null || baseSeed>Int.MAX_VALUE-2) {
    status.text="Stage 3F: enter a valid seed (max ${Int.MAX_VALUE-2})."
    return@setOnClickListener
   }
   val preset=qualityMode.selectedItemPosition
   val seeds=intArrayOf(baseSeed,baseSeed+1,baseSeed+2)
   val names=seeds.map { "xtts_seed_${it}.wav" }

   setGenerationBusy(true)
   lastReport=""
   status.text="Stage 3F: generating 3 variants…"

   Thread {
    val reports=ArrayList<String>()
    var failure:Throwable?=null
    for(i in seeds.indices) {
     val seed=seeds[i]
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
     val result=runCatching {
      XttsSynthesisStage3C(filesDir).run(
       typed,
       sampling,
       names[i]
      ) { message ->
       runOnUiThread {
        status.text="Variant ${i+1}/3 • seed $seed\\n$message"
       }
      }
     }
     if(result.isFailure) {
      failure=result.exceptionOrNull()
      break
     }
     reports.add("VARIANT ${i+1}/3 • seed $seed\\n${result.getOrThrow()}")
    }

    runOnUiThread {
     setGenerationBusy(false)
     if(failure==null) {
      lastReport=reports.joinToString("\\n\\n====================\\n\\n")
      refreshResults(names,names[0])
      copyReport.isEnabled=true
      play.isEnabled=true
      saveWav.isEnabled=true
      shareWav.isEnabled=true
      status.text=buildString {
       append("STAGE 3F SUCCESS — 3 variants saved separately\\n\\n")
       for(i in seeds.indices) append("${i+1}. seed ${seeds[i]} → ${names[i]}\\n")
       append("\\nChoose a result above, then PLAY / SAVE / SHARE.")
      }
     } else {
      lastReport=reports.joinToString("\\n\\n")
      status.text="Stage 3F variant generation failed: ${failure?.javaClass?.simpleName}: ${failure?.message}"
     }
    }
   }.start()
  }

  play.setOnClickListener {
   val selected=resultMode.selectedItem?.toString().orEmpty()
   val wav=File(filesDir,selected)
   if(!wav.exists()) {
    status.text="Generated WAV not found. Run Stage 3C first."
    return@setOnClickListener
   }
   runCatching {
    player?.release()
    player=MediaPlayer().apply {
     setDataSource(wav.absolutePath)
     prepare()
     setOnCompletionListener {
      it.release()
      if(player===it) player=null
     }
     start()
    }
   }.onFailure {
    status.text="Playback failed: ${it.message}"
   }
  }

  saveWav.setOnClickListener {
   val selected=resultMode.selectedItem?.toString().orEmpty()
   val wav=File(filesDir,selected)
   if(!wav.exists()) {
    status.text="Generated WAV not found. Generate a variant first."
    return@setOnClickListener
   }
   pendingSaveName=selected
   startActivityForResult(Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
    addCategory(Intent.CATEGORY_OPENABLE)
    type="audio/wav"
    putExtra(Intent.EXTRA_TITLE,selected)
   },9)
  }

  shareWav.setOnClickListener {
   val selected=resultMode.selectedItem?.toString().orEmpty()
   val wav=File(filesDir,selected)
   if(!wav.exists()) {
    status.text="Generated WAV not found. Run Stage 3C first."
    return@setOnClickListener
   }
   runCatching {
    val uri=FileProvider.getUriForFile(
     this,
     "$packageName.fileprovider",
     wav
    )
    val share=Intent(Intent.ACTION_SEND).apply {
     type="audio/wav"
     putExtra(Intent.EXTRA_STREAM,uri)
     clipData=ClipData.newRawUri("XTTS WAV",uri)
     addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    startActivity(Intent.createChooser(share,"Поделиться WAV"))
   }.onFailure {
    status.text="Share failed: ${it.message}"
   }
  }
 }

 override fun onDestroy() {
  player?.release()
  player=null
  super.onDestroy()
 }

 override fun onActivityResult(requestCode:Int,resultCode:Int,data:Intent?) {
  super.onActivityResult(requestCode,resultCode,data)
  if(resultCode!=Activity.RESULT_OK) return
  val u=data?.data?:return

  if(requestCode==9) {
   val saveName=pendingSaveName
   pendingSaveName=null
   if(saveName.isNullOrBlank()) {
    status.text="WAV save source was lost."
    return
   }
   val wav=File(filesDir,saveName)
   if(!wav.exists()) {
    status.text="Generated WAV not found."
    return
   }
   runCatching {
    contentResolver.openOutputStream(u,"w")!!.use { out ->
     wav.inputStream().use { input -> input.copyTo(out) }
    }
   }.onSuccess {
    status.text="WAV saved successfully."
   }.onFailure {
    status.text="WAV save failed: ${it.message}"
   }
   return
  }

  if(requestCode==7) {
   val f=File(filesDir,"reference.wav")
   contentResolver.openInputStream(u)!!.use { a -> f.outputStream().use { b -> a.copyTo(b) } }
   refPath=f.absolutePath
   File(filesDir,"conditioning_cache").deleteRecursively()
   lastReport=""
   status.text="Reference imported: ${f.length()/1024} KB\nConditioning cache cleared. Run Stage 3A."
  }

  if(requestCode==8) {
   status.text="Importing model archive…"
   Thread {
    val result=runCatching { importModelZip(u); validateModels() }
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
  val models=modelDir.walkTopDown().filter { it.isFile && it.extension.equals("onnx",true) }.toList()
  require(models.isNotEmpty()) { "No .onnx files in imported ZIP" }
  val required=setOf("conditioning_encoder.onnx","speaker_encoder.onnx","gpt_model_int8.onnx","hifigan_vocoder.onnx")
  val missing=required-models.map { it.name }.toSet()
  require(missing.isEmpty()) { "Missing models: ${missing.joinToString()}" }
  val env=OrtEnvironment.getEnvironment()
  val opts=OrtSession.SessionOptions()
  try {
   return models.sortedBy { it.name }.joinToString("\n") { model ->
    env.createSession(model.absolutePath,opts).use { session ->
     "${model.name}: ${session.inputNames.size} inputs / ${session.outputNames.size} outputs"
    }
   }
  } finally { opts.close() }
 }
}
