from pathlib import Path

root = Path(".")
gradle = root / "app/build.gradle.kts"
main = root / "app/src/main/java/com/dorama/xtts/MainActivity.kt"
synth = root / "app/src/main/java/com/dorama/xtts/XttsSynthesisStage3C.kt"

for p in (gradle, main, synth):
    if not p.exists():
        raise SystemExit(f"Missing expected file: {p}")

# ---------- build.gradle.kts ----------
s = gradle.read_text()
assert 'versionCode = 8' in s
assert 'versionName = "0.3.4-stage3e"' in s
s = s.replace('versionCode = 8', 'versionCode = 9')
s = s.replace('versionName = "0.3.4-stage3e"', 'versionName = "0.3.5-stage3f"')
gradle.write_text(s)

# ---------- deterministic sampling ----------
s = synth.read_text()

old = ''' data class Sampling(
  val temperature:Float=0.75f,
  val topK:Int=50,
  val topP:Float=0.85f,
  val repetitionPenalty:Float=10.0f
 )'''
new = ''' data class Sampling(
  val temperature:Float=0.75f,
  val topK:Int=50,
  val topP:Float=0.85f,
  val repetitionPenalty:Float=10.0f,
  val seed:Int?=null
 )'''
assert old in s, "Sampling block not found"
s = s.replace(old, new)

old = '''  val generated=ArrayList<Int>()
  val latents=ArrayList<FloatArray>()
  val used=BooleanArray(1026)
  used[1]=true
  val maxTokens=min(605,melPos.shape[0]-1)'''
new = '''  val generated=ArrayList<Int>()
  val latents=ArrayList<FloatArray>()
  val used=BooleanArray(1026)
  used[1]=true
  val rng=if(sampling.seed!=null) Random(sampling.seed) else Random.Default
  val maxTokens=min(605,melPos.shape[0]-1)'''
assert old in s, "Generation init block not found"
s = s.replace(old, new)

assert '    val token=sampleToken(scores,used,sampling)' in s
s = s.replace(
    '    val token=sampleToken(scores,used,sampling)',
    '    val token=sampleToken(scores,used,sampling,rng)'
)

old = '''   append("Sampling: temp=${sampling.temperature}, topK=${sampling.topK}, topP=${sampling.topP}, rep=${sampling.repetitionPenalty}\\n")
   append("Generated audio tokens: ${generated.size}\\n")'''
new = '''   append("Sampling: temp=${sampling.temperature}, topK=${sampling.topK}, topP=${sampling.topP}, rep=${sampling.repetitionPenalty}\\n")
   append("Seed: ${sampling.seed?.toString() ?: "random"}\\n")
   append("Generated audio tokens: ${generated.size}\\n")'''
assert old in s, "Sampling report block not found"
s = s.replace(old, new)

assert ' private fun sampleToken(raw:FloatArray,used:BooleanArray,cfg:Sampling):Int {' in s
s = s.replace(
    ' private fun sampleToken(raw:FloatArray,used:BooleanArray,cfg:Sampling):Int {',
    ' private fun sampleToken(raw:FloatArray,used:BooleanArray,cfg:Sampling,rng:Random):Int {'
)

assert '  var r=Random.nextDouble()*total' in s
s = s.replace('  var r=Random.nextDouble()*total', '  var r=rng.nextDouble()*total')
synth.write_text(s)

# ---------- Stage 3F UI ----------
s = main.read_text()

old = ''' private var player: MediaPlayer? = null
 private val modelDir by lazy { File(filesDir, "xtts_models") }'''
new = ''' private var player: MediaPlayer? = null
 private var pendingSaveName: String? = null
 private val generatedNames=ArrayList<String>()
 private lateinit var resultMode: Spinner
 private val modelDir by lazy { File(filesDir, "xtts_models") }'''
assert old in s, "MainActivity fields not found"
s = s.replace(old, new)

s = s.replace('text="XTTS-v2 Android V2 • Stage 3E"', 'text="XTTS-v2 Android V2 • Stage 3F"')
s = s.replace(
    'text="Russian XTTS-v2 • Quality Lab + WAV export/share"',
    'text="Russian XTTS-v2 • reproducible seeds + 3 variants"'
)

old = '''   )
   setSelection(0)
  }
  val download=Button(this).apply { text="Import model ZIP / verify ONNX" }'''
new = '''   )
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
  val download=Button(this).apply { text="Import model ZIP / verify ONNX" }'''
assert old in s, "Quality mode tail not found"
s = s.replace(old, new)

old = '''  val synthesize=Button(this).apply { text="Generate voice (Stage 3C)" }
  val play=Button(this).apply {'''
new = '''  val synthesize=Button(this).apply { text="Generate 1 variant (Stage 3F)" }
  val synthesize3=Button(this).apply { text="Generate 3 variants (seed, seed+1, seed+2)" }
  val play=Button(this).apply {'''
assert old in s, "Generate button not found"
s = s.replace(old, new)

old = '''  content.addView(qualityLabel)
  content.addView(qualityMode)
  content.addView(download)
  content.addView(conditioning)
  content.addView(gptTest)
  content.addView(synthesize)
  content.addView(play)'''
new = '''  content.addView(qualityLabel)
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
  content.addView(play)'''
assert old in s, "content add block not found"
s = s.replace(old, new)

s = s.replace(
    'append("Ready for Stage 3A / 3B / 3C / 3D / 3E.")',
    'append("Ready for Stage 3A / 3B / 3C / 3D / 3E / 3F.")'
)

# Stage 3A controls
a = s.index('  conditioning.setOnClickListener {')
b = s.index('  gptTest.setOnClickListener {', a)
section = s[a:b]
section = section.replace(
    '''   synthesize.isEnabled=false
   copyReport.isEnabled=false''',
    '''   synthesize.isEnabled=false
   synthesize3.isEnabled=false
   copyReport.isEnabled=false'''
)
section = section.replace(
    '''     synthesize.isEnabled=true
     result.onSuccess {''',
    '''     synthesize.isEnabled=true
     synthesize3.isEnabled=true
     result.onSuccess {'''
)
s = s[:a] + section + s[b:]

# Stage 3B controls
a = s.index('  gptTest.setOnClickListener {')
b = s.index('  synthesize.setOnClickListener {', a)
section = s[a:b]
section = section.replace(
    '''   synthesize.isEnabled=false
   copyReport.isEnabled=false''',
    '''   synthesize.isEnabled=false
   synthesize3.isEnabled=false
   copyReport.isEnabled=false'''
)
section = section.replace(
    '''     synthesize.isEnabled=true
     result.onSuccess {''',
    '''     synthesize.isEnabled=true
     synthesize3.isEnabled=true
     result.onSuccess {'''
)
s = s[:a] + section + s[b:]

# Replace synthesis handler
start = s.index('  synthesize.setOnClickListener {')
end = s.index('  play.setOnClickListener {', start)

new_handlers = r'''  fun selectedSampling(seed:Int):XttsSynthesisStage3C.Sampling {
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

'''
s = s[:start] + new_handlers + s[end:]

old = '''  play.setOnClickListener {
   val wav=File(filesDir,"xtts_generated.wav")'''
new = '''  play.setOnClickListener {
   val selected=resultMode.selectedItem?.toString().orEmpty()
   val wav=File(filesDir,selected)'''
assert old in s, "play block not found"
s = s.replace(old, new)

old = '''  saveWav.setOnClickListener {
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
  }'''
new = '''  saveWav.setOnClickListener {
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
  }'''
assert old in s, "save block not found"
s = s.replace(old, new)

old = '''  shareWav.setOnClickListener {
   val wav=File(filesDir,"xtts_generated.wav")'''
new = '''  shareWav.setOnClickListener {
   val selected=resultMode.selectedItem?.toString().orEmpty()
   val wav=File(filesDir,selected)'''
assert old in s, "share block not found"
s = s.replace(old, new)

old = '''  if(requestCode==9) {
   val wav=File(filesDir,"xtts_generated.wav")
   if(!wav.exists()) {
    status.text="Generated WAV not found."
    return
   }'''
new = '''  if(requestCode==9) {
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
   }'''
assert old in s, "save result block not found"
s = s.replace(old, new)

main.write_text(s)

print("Stage 3F applied successfully")
print("Stage 3D save/share retained")
print("Stage 3E presets retained; default is Original 3C")
print("Added deterministic seed")
print("Added Generate 1 variant")
print("Added Generate 3 variants: seed / seed+1 / seed+2")
print("Each result is stored separately and selectable for Play / Save / Share")
print("Version: code 9 / 0.3.5-stage3f")
