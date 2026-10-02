from pathlib import Path

root = Path(".")
main = root / "app/src/main/java/com/dorama/xtts/MainActivity.kt"
gradle = root / "app/build.gradle.kts"
manifest = root / "app/src/main/AndroidManifest.xml"
xml_dir = root / "app/src/main/res/xml"
paths_xml = xml_dir / "file_paths.xml"

for p in (main, gradle, manifest):
    if not p.exists():
        raise SystemExit(f"Missing required file: {p}")

ms = main.read_text()
gr = gradle.read_text()
mf = manifest.read_text()

if "Stage 3D" in ms or 'versionName = "0.3.3-stage3d"' in gr:
    raise SystemExit("Stage 3D appears to be already applied.")

def replace_once(text, old, new, label):
    if old not in text:
        raise SystemExit(f"Stage 3D patch stopped: cannot find {label}. No files were changed.")
    return text.replace(old, new, 1)

ms2 = ms
ms2 = replace_once(
    ms2,
    "import android.os.Bundle\nimport android.widget.*",
    "import android.os.Bundle\nimport android.text.InputType\nimport android.view.Gravity\nimport android.widget.*",
    "Android UI imports",
)
ms2 = replace_once(
    ms2,
    "import androidx.appcompat.app.AppCompatActivity\n",
    "import androidx.appcompat.app.AppCompatActivity\nimport androidx.core.content.FileProvider\n",
    "FileProvider import",
)

ms2 = replace_once(
    ms2,
    'text="XTTS-v2 Android V2 • Stage 3C"',
    'text="XTTS-v2 Android V2 • Stage 3D"',
    "Stage title",
)
ms2 = replace_once(
    ms2,
    'text="Russian XTTS-v2 • full local synthesis to WAV"',
    'text="Russian XTTS-v2 • local voice + WAV export/share"',
    "Stage subtitle",
)

old_input = '''  val inputText=EditText(this).apply {
   hint="Русский текст"
   setText("Сегодня я хочу рассказать вам об одной удивительной истории.")
  }'''
new_input = '''  val inputText=EditText(this).apply {
   hint="Русский текст"
   setText("Сегодня я хочу рассказать вам об одной удивительной истории.")
   minLines=4
   gravity=Gravity.TOP or Gravity.START
   inputType=InputType.TYPE_CLASS_TEXT or
    InputType.TYPE_TEXT_FLAG_MULTI_LINE or
    InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
   setHorizontallyScrolling(false)
  }'''
ms2 = replace_once(ms2, old_input, new_input, "text editor")

old_buttons = '''  val play=Button(this).apply {
   text="Play generated WAV"
   isEnabled=File(filesDir,"xtts_generated.wav").exists()
  }
  val copyReport=Button(this).apply {'''
new_buttons = '''  val play=Button(this).apply {
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
  val copyReport=Button(this).apply {'''
ms2 = replace_once(ms2, old_buttons, new_buttons, "WAV buttons")

ms2 = replace_once(
    ms2,
    '''  content.addView(synthesize)
  content.addView(play)
  content.addView(copyReport)''',
    '''  content.addView(synthesize)
  content.addView(play)
  content.addView(saveWav)
  content.addView(shareWav)
  content.addView(copyReport)''',
    "WAV button layout",
)

ms2 = replace_once(
    ms2,
    'append("Ready for Stage 3A / 3B / 3C.")',
    'append("Ready for Stage 3A / 3B / 3C / 3D.")',
    "ready status",
)

old_start = '''   synthesize.isEnabled=false
   play.isEnabled=false
   copyReport.isEnabled=false
   lastReport=""'''
new_start = '''   synthesize.isEnabled=false
   play.isEnabled=false
   saveWav.isEnabled=false
   shareWav.isEnabled=false
   copyReport.isEnabled=false
   lastReport=""'''
ms2 = replace_once(ms2, old_start, new_start, "synthesis start controls")

old_success = '''      copyReport.isEnabled=true
      play.isEnabled=File(filesDir,"xtts_generated.wav").exists()
      status.text=it'''
new_success = '''      copyReport.isEnabled=true
      val generated=File(filesDir,"xtts_generated.wav").exists()
      play.isEnabled=generated
      saveWav.isEnabled=generated
      shareWav.isEnabled=generated
      status.text=it'''
ms2 = replace_once(ms2, old_success, new_success, "synthesis success controls")

old_failure = '''      copyReport.isEnabled=false
      play.isEnabled=File(filesDir,"xtts_generated.wav").exists()
      status.text="Stage 3C failed: ${it.javaClass.simpleName}: ${it.message}"'''
new_failure = '''      copyReport.isEnabled=false
      val generated=File(filesDir,"xtts_generated.wav").exists()
      play.isEnabled=generated
      saveWav.isEnabled=generated
      shareWav.isEnabled=generated
      status.text="Stage 3C failed: ${it.javaClass.simpleName}: ${it.message}"'''
ms2 = replace_once(ms2, old_failure, new_failure, "synthesis failure controls")

old_play_end = '''  play.setOnClickListener {
   val wav=File(filesDir,"xtts_generated.wav")
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
 }'''

new_play_end = '''  play.setOnClickListener {
   val wav=File(filesDir,"xtts_generated.wav")
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
   val wav=File(filesDir,"xtts_generated.wav")
   if(!wav.exists()) {
    status.text="Generated WAV not found. Run Stage 3C first."
    return@setOnClickListener
   }
   startActivityForResult(Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
    addCategory(Intent.CATEGORY_OPENABLE)
    type="audio/wav"
    putExtra(Intent.EXTRA_TITLE,"xtts_generated.wav")
   },9)
  }

  shareWav.setOnClickListener {
   val wav=File(filesDir,"xtts_generated.wav")
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
 }'''
ms2 = replace_once(ms2, old_play_end, new_play_end, "playback handler")

old_result = '''  if(requestCode==7) {
   val f=File(filesDir,"reference.wav")'''
new_result = '''  if(requestCode==9) {
   val wav=File(filesDir,"xtts_generated.wav")
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
   val f=File(filesDir,"reference.wav")'''
ms2 = replace_once(ms2, old_result, new_result, "save result handler")

gr2 = replace_once(gr, "versionCode = 6", "versionCode = 7", "versionCode")
gr2 = replace_once(
    gr2,
    'versionName = "0.3.2-stage3c"',
    'versionName = "0.3.3-stage3d"',
    "versionName",
)

provider = '''  <provider
   android:name="androidx.core.content.FileProvider"
   android:authorities="${applicationId}.fileprovider"
   android:exported="false"
   android:grantUriPermissions="true">
   <meta-data
    android:name="android.support.FILE_PROVIDER_PATHS"
    android:resource="@xml/file_paths"/>
  </provider>
'''
if "androidx.core.content.FileProvider" in mf:
    raise SystemExit("Stage 3D patch stopped: FileProvider already exists. No files were changed.")
mf2 = replace_once(mf, " </application>", provider + " </application>", "application closing tag")

main.write_text(ms2)
gradle.write_text(gr2)
manifest.write_text(mf2)
xml_dir.mkdir(parents=True, exist_ok=True)
paths_xml.write_text('''<?xml version="1.0" encoding="utf-8"?>
<paths xmlns:android="http://schemas.android.com/apk/res/android">
 <files-path name="generated_audio" path="."/>
</paths>
''')

print("Stage 3D applied successfully")
print("Stage 3C synthesis engine retained unchanged")
print("Added: multiline Russian text editor")
print("Added: SAVE WAV via Android document picker")
print("Added: SHARE WAV via FileProvider")
print("New file: app/src/main/res/xml/file_paths.xml")
print("Version: code 7 / 0.3.3-stage3d")
