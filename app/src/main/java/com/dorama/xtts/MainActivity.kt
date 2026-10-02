package com.dorama.xtts

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
   text="XTTS-v2 Android V2 • Stage 3B"
   textSize=24f
  })
  content.addView(TextView(this).apply {
   text="Russian BPE → GPT INT8 prefill + KV-cache decode"
  })

  val import=Button(this).apply { text="Import voice WAV (3–6 sec)" }
  val inputText=EditText(this).apply {
   hint="Русский текст"
   setText("Сегодня я хочу рассказать вам об одной удивительной истории.")
  }
  val download=Button(this).apply { text="Import model ZIP / verify ONNX" }
  val conditioning=Button(this).apply { text="Compute voice conditioning (Stage 3A)" }
  val gptTest=Button(this).apply { text="Test Russian GPT (Stage 3B)" }
  val copyReport=Button(this).apply {
   text="Copy report"
   isEnabled=false
  }
  status=TextView(this).apply { setTextIsSelectable(true) }

  content.addView(import)
  content.addView(inputText)
  content.addView(download)
  content.addView(conditioning)
  content.addView(gptTest)
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
     append("Ready for Stage 3A / 3B.")
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
 }

 override fun onActivityResult(requestCode:Int,resultCode:Int,data:Intent?) {
  super.onActivityResult(requestCode,resultCode,data)
  if(resultCode!=Activity.RESULT_OK) return
  val u=data?.data?:return

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
