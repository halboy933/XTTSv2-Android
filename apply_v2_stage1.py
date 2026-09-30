from pathlib import Path
p=Path('app/src/main/java/com/dorama/xtts/MainActivity.kt')
s=p.read_text(encoding='utf-8')
if 'XTTS_V2_STAGE1' in s:
    raise SystemExit('Already patched')
assert 'class MainActivity : AppCompatActivity()' in s
assert 'download.setOnClickListener{status.text="V1 loader manifest ready.' in s
s=s.replace('import java.io.File','import java.io.File\nimport java.util.zip.ZipInputStream\nimport ai.onnxruntime.OrtSession')
s=s.replace(' private var refPath: String? = null',' private var refPath: String? = null\n private val modelDir by lazy { File(filesDir, "xtts_models") }')
s=s.replace('  download.setOnClickListener{status.text="V1 loader manifest ready. Full XTTS inference engine is the next build stage; model package is not bundled in APK."}', '''  // XTTS_V2_STAGE1: import model archive; verify that ONNX sessions open.
  download.text = "Import model ZIP / verify ONNX"
  download.setOnClickListener {
    startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
      type = "application/zip"
      addCategory(Intent.CATEGORY_OPENABLE)
    }, 8)
  }''')
s=s.replace('  generate.setOnClickListener{Toast.makeText(this,"Inference engine not activated in bootstrap V1",Toast.LENGTH_LONG).show()}', '''  generate.setOnClickListener { status.text = "Full XTTS inference is not implemented in stage 1" }''')
start=s.index(' override fun onActivityResult(')
s=s[:start]+''' override fun onActivityResult(requestCode:Int,resultCode:Int,data:Intent?) {
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
          val name=entry.name.replace('\\\\','/')
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
  val env=OrtEnvironment.getEnvironment()
  val opts=OrtSession.SessionOptions()
  try {
    val results=mutableListOf<String>()
    for(model in models) {
      env.createSession(model.absolutePath,opts).use { session ->
        results.add("${model.name}: ${session.inputNames.size} inputs / ${session.outputNames.size} outputs")
      }
    }
    return results.joinToString("\\n")
  } finally { opts.close() }
 }
}
'''
p.write_text(s,encoding='utf-8')
print('Stage 1 patch applied to',p)
