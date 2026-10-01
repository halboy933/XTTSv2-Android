package com.dorama.xtts

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.widget.*
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import androidx.appcompat.app.AppCompatActivity
import ai.onnxruntime.OrtEnvironment
import java.io.File
import java.util.zip.ZipInputStream
import ai.onnxruntime.OrtSession

class MainActivity : AppCompatActivity() {
 private lateinit var status: TextView
 private var refPath: String? = null
 private var lastOnnxReport: String = ""
 private val modelDir by lazy { File(filesDir, "xtts_models") }
 override fun onCreate(savedInstanceState: Bundle?) {
  super.onCreate(savedInstanceState)
  val content=LinearLayout(this).apply{
   orientation=LinearLayout.VERTICAL
   setPadding(32,32,32,32)
  }
  val scroll=ScrollView(this).apply{
   isFillViewport=true
   addView(content)
  }
  val root=content
  root.addView(TextView(this).apply{text="XTTS-v2 Android V2 • Stage 2.1";textSize=24f})
  root.addView(TextView(this).apply{text="Russian • ONNX Runtime • INT8 GPT"})
  val import=Button(this).apply{text="Import voice WAV (3–6 sec)"}
  val text=EditText(this).apply{hint="Русский текст";setText("Сегодня я хочу рассказать вам об одной удивительной истории.")}
  val download = Button(this).apply { setText("Download / verify XTTS models") }
  val generate = Button(this).apply { setText("Inspect ONNX interfaces"); isEnabled = true }
  status = TextView(this).apply {
   setText("Status: ONNX Runtime check…")
   setTextIsSelectable(true)
  }
  val copyReport = Button(this).apply {
   setText("Copy ONNX info")
   isEnabled = false
  }
  root.addView(import)
  root.addView(text)
  root.addView(download)
  root.addView(generate)
  root.addView(copyReport)
  root.addView(status)
  setContentView(scroll)

  copyReport.setOnClickListener {
   if(lastOnnxReport.isNotBlank()) {
    val clipboard=getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
    clipboard.setPrimaryClip(ClipData.newPlainText("XTTS ONNX interfaces",lastOnnxReport))
    Toast.makeText(this,"ONNX info copied",Toast.LENGTH_SHORT).show()
   }
  }
  runCatching { OrtEnvironment.getEnvironment() }.onSuccess{status.text="Status: ONNX Runtime OK. Import reference and download models."}.onFailure{status.text="ONNX Runtime error: ${it.message}"}
  import.setOnClickListener{startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply{type="audio/*";addCategory(Intent.CATEGORY_OPENABLE)},7)}
  // XTTS_V2_STAGE1: import model archive; verify that ONNX sessions open.
  download.text = "Import model ZIP / verify ONNX"
  download.setOnClickListener {
    startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
      type = "application/zip"
      addCategory(Intent.CATEGORY_OPENABLE)
    }, 8)
  }
  generate.setOnClickListener {
    status.text = "Opening ONNX models..."
    Thread {
      val result = runCatching { inspectModels() }
      runOnUiThread {
        status.text = result.fold(
          {
            lastOnnxReport=it
            copyReport.isEnabled=true
            it
          },
          {
            lastOnnxReport=""
            copyReport.isEnabled=false
            "ONNX inspect failed: ${it.message}"
          }
        )
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
    status.text="Reference imported: ${f.length()/1024} KB"
  }
  if(requestCode==8) {
    status.text="Importing model archive..."
    Thread {
      val result=runCatching { importModelZip(u); validateModels() }
      runOnUiThread { status.text=result.fold({"ONNX check: $it"},{"ONNX check failed: ${it.message}"}) }
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


 private fun inspectModels():String {
  val models=modelDir.walkTopDown()
    .filter { it.isFile && it.extension.equals("onnx",true) }
    .toList()

  require(models.isNotEmpty()) {
    "Models are missing. Import model ZIP first."
  }

  val env=OrtEnvironment.getEnvironment()
  val opts=OrtSession.SessionOptions()

  return try {
    buildString {
      models.sortedBy { it.name }.forEach { model ->

        env.createSession(model.absolutePath,opts).use { session ->

          append("\n[")
          append(model.name)
          append("]\n")

          append("IN:\n")

          session.inputInfo.entries.forEach { (name,info) ->
            append("  ")
            append(name)
            append(" : ")
            append(info.info.toString())
            append("\n")
          }

          append("OUT:\n")

          session.outputInfo.entries.forEach { (name,info) ->
            append("  ")
            append(name)
            append(" : ")
            append(info.info.toString())
            append("\n")
          }
        }
      }
    }
  } finally {
    opts.close()
  }
 }

 private fun validateModels():String {
  val models=modelDir.walkTopDown().filter { it.isFile && it.extension.equals("onnx",true) }.toList()
  require(models.isNotEmpty()) { "No .onnx files in imported ZIP" }
  val env=OrtEnvironment.getEnvironment()
  val opts=OrtSession.SessionOptions()
  try {
    val results=mutableListOf<String>()
    for(model in models) {
      env.createSession(model.absolutePath,opts).use { session ->
        results.add("${model.name}: ${session.inputNames.size} inputs / ${session.outputNames.size} outputs")
      }
    }
    return results.joinToString("\n")
  } finally { opts.close() }
 }
}
