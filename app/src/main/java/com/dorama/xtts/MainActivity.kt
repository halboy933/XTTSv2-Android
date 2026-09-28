package com.dorama.xtts

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import ai.onnxruntime.OrtEnvironment
import java.io.File

class MainActivity : AppCompatActivity() {
 private lateinit var status: TextView
 private var refPath: String? = null
 override fun onCreate(savedInstanceState: Bundle?) {
  super.onCreate(savedInstanceState)
  val root=LinearLayout(this).apply{orientation=LinearLayout.VERTICAL;setPadding(32,32,32,32)}
  root.addView(TextView(this).apply{text="XTTS-v2 Android V1";textSize=24f})
  root.addView(TextView(this).apply{text="Russian • ONNX Runtime • INT8 GPT"})
  val import=Button(this).apply{text="Import voice WAV (3–6 sec)"}
  val text=EditText(this).apply{hint="Русский текст";setText("Сегодня я хочу рассказать вам об одной удивительной истории.")}
  val download = Button(this).apply { setText("Download / verify XTTS models") }
  val generate = Button(this).apply { setText("Generate WAV"); isEnabled = false }
  status = TextView(this).apply { setText("Status: ONNX Runtime check…") }
  root.addView(import);root.addView(text);root.addView(download);root.addView(generate);root.addView(status)
  setContentView(root)
  runCatching { OrtEnvironment.getEnvironment() }.onSuccess{status.text="Status: ONNX Runtime OK. Import reference and download models."}.onFailure{status.text="ONNX Runtime error: ${it.message}"}
  import.setOnClickListener{startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply{type="audio/*";addCategory(Intent.CATEGORY_OPENABLE)},7)}
  download.setOnClickListener{status.text="V1 loader manifest ready. Full XTTS inference engine is the next build stage; model package is not bundled in APK."}
  generate.setOnClickListener{Toast.makeText(this,"Inference engine not activated in bootstrap V1",Toast.LENGTH_LONG).show()}
 }
 override fun onActivityResult(requestCode:Int,resultCode:Int,data:Intent?){super.onActivityResult(requestCode,resultCode,data);if(requestCode==7&&resultCode==Activity.RESULT_OK){val u=data?.data?:return;val f=File(filesDir,"reference.wav");contentResolver.openInputStream(u)!!.use{a->f.outputStream().use{b->a.copyTo(b)}};refPath=f.absolutePath;status.text="Reference imported: ${f.length()/1024} KB"}}
}
