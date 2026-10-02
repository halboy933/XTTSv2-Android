#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GRADLE = ROOT / "app/build.gradle.kts"
MAIN = ROOT / "app/src/main/java/com/dorama/xtts/MainActivity.kt"
SYNTH = ROOT / "app/src/main/java/com/dorama/xtts/XttsSynthesisStage3C.kt"

for p in (GRADLE, MAIN, SYNTH):
    if not p.exists():
        raise SystemExit(f"Missing required file: {p}")

def replace_once(path: Path, old: str, new: str, label: str):
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly 1 match, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")

replace_once(
    GRADLE,
    'versionCode = 7\n        versionName = "0.3.3-stage3d"',
    'versionCode = 8\n        versionName = "0.3.4-stage3e"',
    "Gradle version"
)

replace_once(
    SYNTH,
    ''' fun run(text:String,progress:(String)->Unit):String {
  val totalStart=System.nanoTime()''',
    ''' fun run(text:String,progress:(String)->Unit):String =
  run(text,Sampling(),"xtts_generated.wav",progress)

 fun run(
  text:String,
  sampling:Sampling,
  outputName:String="xtts_generated.wav",
  progress:(String)->Unit
 ):String {
  require(outputName.matches(Regex("[A-Za-z0-9._-]+"))) { "Invalid output file name" }
  val totalStart=System.nanoTime()''',
    "Synthesis configurable overload"
)

replace_once(
    SYNTH,
    '''  val sampling=Sampling()
  val generated=ArrayList<Int>()''',
    '''  val generated=ArrayList<Int>()''',
    "Remove hardcoded sampling"
)

replace_once(
    SYNTH,
    '''  val outFile=File(filesDir,"xtts_generated.wav")''',
    '''  val outFile=File(filesDir,outputName)''',
    "Configurable output name"
)

replace_once(
    SYNTH,
    '''   append("Generated audio tokens: ${generated.size}\\n")
   append("Stop token reached: $stoppedByToken\\n")''',
    '''   append("Sampling: temp=${sampling.temperature}, topK=${sampling.topK}, topP=${sampling.topP}, rep=${sampling.repetitionPenalty}\\n")
   append("Generated audio tokens: ${generated.size}\\n")
   append("Stop token reached: $stoppedByToken\\n")''',
    "Sampling report"
)

replace_once(
    MAIN,
    '''   text="XTTS-v2 Android V2 • Stage 3D"
   textSize=24f
  })
  content.addView(TextView(this).apply {
   text="Russian XTTS-v2 • local voice + WAV export/share"
  })''',
    '''   text="XTTS-v2 Android V2 • Stage 3E"
   textSize=24f
  })
  content.addView(TextView(this).apply {
   text="Russian XTTS-v2 • Quality Lab + WAV export/share"
  })''',
    "Stage title"
)

replace_once(
    MAIN,
    '''  val download=Button(this).apply { text="Import model ZIP / verify ONNX" }''',
    '''  val qualityLabel=TextView(this).apply {
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
   setSelection(1)
  }
  val download=Button(this).apply { text="Import model ZIP / verify ONNX" }''',
    "Quality UI declaration"
)

replace_once(
    MAIN,
    '''  content.addView(import)
  content.addView(inputText)
  content.addView(download)''',
    '''  content.addView(import)
  content.addView(inputText)
  content.addView(qualityLabel)
  content.addView(qualityMode)
  content.addView(download)''',
    "Quality UI layout"
)

replace_once(
    MAIN,
    '''     append("Ready for Stage 3A / 3B / 3C / 3D.")''',
    '''     append("Ready for Stage 3A / 3B / 3C / 3D / 3E.")''',
    "Ready status"
)

replace_once(
    MAIN,
    '''   conditioning.isEnabled=false
   gptTest.isEnabled=false
   synthesize.isEnabled=false
   play.isEnabled=false
   saveWav.isEnabled=false''',
    '''   conditioning.isEnabled=false
   gptTest.isEnabled=false
   synthesize.isEnabled=false
   qualityMode.isEnabled=false
   play.isEnabled=false
   saveWav.isEnabled=false''',
    "Disable quality selector during synthesis"
)

replace_once(
    MAIN,
    '''    val result=runCatching {
     XttsSynthesisStage3C(filesDir).run(typed) { message ->
      runOnUiThread { status.text=message }
     }
    }''',
    '''    val result=runCatching {
     val preset=qualityMode.selectedItemPosition
     val sampling=when(preset) {
      1 -> XttsSynthesisStage3C.Sampling(
       temperature=0.65f,
       topK=30,
       topP=0.90f,
       repetitionPenalty=10.0f
      )
      else -> XttsSynthesisStage3C.Sampling()
     }
     XttsSynthesisStage3C(filesDir).run(
      typed,
      sampling,
      "xtts_generated.wav"
     ) { message ->
      runOnUiThread { status.text=message }
     }
    }''',
    "Quality sampling call"
)

text = MAIN.read_text(encoding="utf-8")
needle = '''    runOnUiThread {
     conditioning.isEnabled=true
     gptTest.isEnabled=true
     synthesize.isEnabled=true
     result.onSuccess {
      lastReport=it
      copyReport.isEnabled=true
      val generated=File(filesDir,"xtts_generated.wav").exists()'''
if needle in text:
    text = text.replace(
        needle,
        '''    runOnUiThread {
     conditioning.isEnabled=true
     gptTest.isEnabled=true
     synthesize.isEnabled=true
     qualityMode.isEnabled=true
     result.onSuccess {
      lastReport=it
      copyReport.isEnabled=true
      val generated=File(filesDir,"xtts_generated.wav").exists()''',
        1
    )
    MAIN.write_text(text, encoding="utf-8")
elif '''qualityMode.isEnabled=true
     result.onSuccess {
      lastReport=it
      copyReport.isEnabled=true
      val generated=File(filesDir,"xtts_generated.wav").exists()''' not in text:
    raise SystemExit("Could not verify qualityMode re-enable in synthesis result block")

print("Stage 3E applied successfully")
print("Stage 3D save/share retained")
print("Stage 3C default synthesis behavior retained")
print("Added Quality Lab selector:")
print("  Original 3C: temp 0.75 / topK 50 / topP 0.85 / rep 10")
print("  Clearer 3E:  temp 0.65 / topK 30 / topP 0.90 / rep 10")
print("Default UI selection: Clearer 3E")
print("Version: code 8 / 0.3.4-stage3e")
