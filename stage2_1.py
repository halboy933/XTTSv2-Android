from pathlib import Path

p = Path("app/src/main/java/com/dorama/xtts/MainActivity.kt")
s = p.read_text()

s = s.replace(
    'import android.widget.*',
    '''import android.widget.*
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context'''
)

s = s.replace(
    'private var refPath: String? = null',
    '''private var refPath: String? = null
 private var lastOnnxReport: String = ""'''
)

s = s.replace(
    'val root=LinearLayout(this).apply{orientation=LinearLayout.VERTICAL;setPadding(32,32,32,32)}',
    '''val content=LinearLayout(this).apply{
   orientation=LinearLayout.VERTICAL
   setPadding(32,32,32,32)
  }
  val scroll=ScrollView(this).apply{
   isFillViewport=true
   addView(content)
  }
  val root=content'''
)

s = s.replace(
    'text="XTTS-v2 Android V2 • Stage 2"',
    'text="XTTS-v2 Android V2 • Stage 2.1"'
)

s = s.replace(
    'status = TextView(this).apply { setText("Status: ONNX Runtime check…") }',
    '''status = TextView(this).apply {
   setText("Status: ONNX Runtime check…")
   setTextIsSelectable(true)
  }
  val copyReport = Button(this).apply {
   setText("Copy ONNX info")
   isEnabled = false
  }'''
)

s = s.replace(
    'root.addView(import);root.addView(text);root.addView(download);root.addView(generate);root.addView(status)\n  setContentView(root)',
    '''root.addView(import)
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
  }'''
)

s = s.replace(
    '''status.text = result.fold(
          { it },
          { "ONNX inspect failed: ${it.message}" }
        )''',
    '''status.text = result.fold(
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
        )'''
)

p.write_text(s)

g = Path("app/build.gradle.kts")
gs = g.read_text()
gs = gs.replace('versionCode = 2', 'versionCode = 3')
gs = gs.replace('versionName = "0.2.1"', 'versionName = "0.2.1-stage2.1"')
g.write_text(gs)

print("Stage 2.1 applied successfully")
print("MainActivity: scrolling + selectable report + COPY ONNX INFO")
print("Version: code 3 / 0.2.1-stage2.1")