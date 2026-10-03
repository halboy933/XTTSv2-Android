from pathlib import Path

root = Path(".")
gradle = root / "app/build.gradle.kts"
main = root / "app/src/main/java/com/dorama/xtts/MainActivity.kt"
dialog = root / "app/src/main/java/com/dorama/xtts/GenerationProgressDialog.kt"
engine = root / "app/src/main/java/com/dorama/xtts/XttsSmoothHybridSearch.kt"

for p in (gradle, main, dialog):
    if not p.exists():
        raise SystemExit(f"Missing expected file: {p}")

# ---------- version ----------
s = gradle.read_text()
assert 'versionCode = 13' in s, "Expected Stage 3J versionCode 13"
assert 'versionName = "0.3.9-stage3j"' in s, "Expected Stage 3J versionName"
s = s.replace('versionCode = 13', 'versionCode = 14')
s = s.replace('versionName = "0.3.9-stage3j"', 'versionName = "0.3.10-stage3k"')
gradle.write_text(s)

# ---------- dialog: optional manual lock button ----------
s = dialog.read_text()

field_marker = ' private val playButton=Button(activity)\n'
assert field_marker in s, "Dialog playButton field marker not found"
s = s.replace(
    field_marker,
    field_marker + ' private val lockButton=Button(activity)\n',
    1
)

close_init = '''  closeButton.apply {
   text="Закрыть"
   isEnabled=false
  }
'''
assert close_init in s, "Dialog closeButton init marker not found"
lock_init = '''  lockButton.apply {
   text="✓ Закрепить выбранный профиль"
   visibility=View.GONE
  }
'''
s = s.replace(close_init, lock_init + close_init, 1)

add_marker = '  content.addView(playButton)\n'
assert add_marker in s, "Dialog addView play marker not found"
s = s.replace(
    add_marker,
    add_marker + '  content.addView(lockButton)\n',
    1
)

signature = ' fun complete(results:List<ResultItem>) {\n'
assert signature in s, "Dialog complete signature not found"
s = s.replace(
    signature,
    ' fun complete(results:List<ResultItem>,onLock:((ResultItem)->Unit)?=null) {\n',
    1
)

visibility_marker = '''  playButton.visibility=View.VISIBLE
  closeButton.isEnabled=true
'''
assert visibility_marker in s, "Dialog completion visibility marker not found"
replacement = '''  playButton.visibility=View.VISIBLE
  closeButton.isEnabled=true
  lockButton.visibility=if(onLock!=null) View.VISIBLE else View.GONE
  lockButton.isEnabled=true
'''
s = s.replace(visibility_marker, replacement, 1)

selection_marker = '''  resultSpinner.setSelection(0)
 }

 fun fail(text:String) {
'''
assert selection_marker in s, "Dialog completion end marker not found"
lock_logic = '''  resultSpinner.setSelection(0)

  lockButton.setOnClickListener {
   val pos=resultSpinner.selectedItemPosition
   if(pos !in items.indices || onLock==null) return@setOnClickListener
   val chosen=items[pos]
   runCatching {
    onLock(chosen)
   }.onSuccess {
    lockButton.isEnabled=false
    lockButton.text="✓ Профиль закреплён"
    val name=chosen.displayName ?: "seed ${chosen.seed}"
    message.text=message.text.toString()+"\\n✓ Закреплён: $name"
   }.onFailure {
    message.text="Ошибка закрепления профиля: ${it.message}"
   }
  }
 }

 fun fail(text:String) {
'''
s = s.replace(selection_marker, lock_logic, 1)
dialog.write_text(s)

# ---------- new Stage 3K engine ----------
engine.write_text(r'''package com.dorama.xtts

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale

class XttsSmoothHybridSearch(private val filesDir:File) {
 data class Candidate(
  val id:String,
  val label:String,
  val condWeight4:Float,
  val condWeight5:Float,
  val generatedFile:File,
  val similarity:Double,
  val cacheSnapshot:File
 )

 data class SearchResult(
  val candidates:List<Candidate>,
  val ranked:List<Candidate>,
  val report:String
 )

 private val cacheDir=File(filesDir,"conditioning_cache")
 private val workRoot=File(filesDir,"smooth_hybrid_candidates")
 private val previousCache=File(filesDir,"smooth_hybrid_previous_cache")

 fun run(
  longReference:File,
  text:String,
  preset:Int,
  seed:Int=12346,
  progress:(String)->Unit
 ):SearchResult {
  require(longReference.exists()) { "reference_long.wav is missing." }
  require(text.isNotBlank()) { "Enter Russian test text first." }

  val conditioning=XttsConditioning(filesDir)
  val similarityEngine=XttsVoiceSimilarity(filesDir)

  progress("Stage 3K 1/5: preparing Profiles 4 and 5…")
  val segments=conditioning.createReferenceSegments(longReference,5.5,5)
  require(segments.size>=5) { "Stage 3K requires 5 reference profiles." }

  progress("Stage 3K 2/5: building stable speaker target…")
  val targetVectors=segments.mapIndexed { i,segment ->
   progress("Stage 3K 2/5: speaker target ${i+1}/5…")
   conditioning.computeSpeakerEmbedding(segment.file)
  }
  val target=weightedAverage(targetVectors,FloatArray(targetVectors.size){1f/targetVectors.size},512)

  previousCache.deleteRecursively()
  if(cacheDir.exists()) cacheDir.copyRecursively(previousCache,overwrite=true)
  workRoot.deleteRecursively()
  workRoot.mkdirs()

  try {
   // Compute Profile 4 and Profile 5 caches once.
   val baseCond=HashMap<Int,FloatArray>()
   val baseSpeaker=HashMap<Int,FloatArray>()
   for(profileIndex in intArrayOf(4,5)) {
    val segment=segments[profileIndex-1]
    progress("Stage 3K 3/5: extracting Profile $profileIndex conditioning…")
    conditioning.run(segment.file) { message ->
     progress("Stage 3K 3/5 • Profile $profileIndex\n$message")
    }
    baseCond[profileIndex]=readFloat32LE(File(cacheDir,"cond_latents.f32"))
    baseSpeaker[profileIndex]=readFloat32LE(File(cacheDir,"speaker_embedding.f32"))
   }

   val cond4=baseCond.getValue(4)
   val cond5=baseCond.getValue(5)
   val speaker4=baseSpeaker.getValue(4)
   val speaker5=baseSpeaker.getValue(5)
   require(cond4.size==32*1024 && cond5.size==32*1024)
   require(speaker4.size==512 && speaker5.size==512)

   // Keep the successful 4+5 speaker timbre in every candidate.
   val speaker45=weightedAverage(
    listOf(speaker4,speaker5),
    floatArrayOf(0.5f,0.5f),
    512
   )

   data class Config(
    val id:String,
    val label:String,
    val w4:Float,
    val w5:Float
   )

   val configs=listOf(
    Config("p45_base","A • 4+5 baseline",0.50f,0.50f),
    Config("p4_speaker45","B • cond 4 + speaker 4+5",1.00f,0.00f),
    Config("p4_75_p5_25","C • cond 75% 4 + 25% 5",0.75f,0.25f),
    Config("p4_85_p5_15","D • cond 85% 4 + 15% 5",0.85f,0.15f)
   )

   val candidates=ArrayList<Candidate>()

   for((index,cfg) in configs.withIndex()) {
    progress(
     "Stage 3K 4/5: вариант ${index+1}/${configs.size}\n"+
      "${cfg.label}\nГотовлю hybrid conditioning…"
    )

    val cond=weightedAverage(
     listOf(cond4,cond5),
     floatArrayOf(cfg.w4,cfg.w5),
     32*1024
    )

    cacheDir.deleteRecursively()
    cacheDir.mkdirs()
    writeFloat32LE(File(cacheDir,"cond_latents.f32"),cond)
    writeFloat32LE(File(cacheDir,"speaker_embedding.f32"),speaker45)
    File(cacheDir,"meta.txt").writeText(
     "stage=3K\n"+
      "label=${cfg.label}\n"+
      "cond_profile4=${cfg.w4}\n"+
      "cond_profile5=${cfg.w5}\n"+
      "speaker_profile4=0.5\n"+
      "speaker_profile5=0.5\n"+
      "seed=$seed\n"
    )

    val snapshot=File(workRoot,cfg.id)
    snapshot.deleteRecursively()
    snapshot.mkdirs()
    cacheDir.copyRecursively(File(snapshot,"conditioning_cache"),overwrite=true)

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

    val generated=File(filesDir,"xtts_smooth_${cfg.id}_seed_${seed}.wav")
    XttsSynthesisStage3C(filesDir).run(
     text,
     sampling,
     generated.name
    ) { message ->
     progress("Stage 3K 4/5 • ${cfg.label}\n$message")
    }

    progress("Stage 3K 4/5 • ${cfg.label}\nОцениваю сходство голоса…")
    val score=similarityEngine.scoreAgainstEmbedding(target,generated)

    candidates.add(
     Candidate(
      id=cfg.id,
      label=cfg.label,
      condWeight4=cfg.w4,
      condWeight5=cfg.w5,
      generatedFile=generated,
      similarity=score,
      cacheSnapshot=File(snapshot,"conditioning_cache")
     )
    )
   }

   // Important: do not auto-lock by similarity. Restore Stage 3J until user listens.
   cacheDir.deleteRecursively()
   if(previousCache.exists()) previousCache.copyRecursively(cacheDir,overwrite=true)

   progress("Stage 3K 5/5: готово. Прослушайте варианты и закрепите лучший вручную.")
   val ranked=candidates.sortedByDescending { it.similarity }

   val report=buildString {
    append("STAGE 3K SUCCESS — smooth 4+5 hybrids\n\n")
    append("Fixed test seed: $seed\n")
    append("Speaker embedding: 50% Profile 4 + 50% Profile 5 for every variant\n")
    append("Nothing is auto-locked: choose by ear.\n\n")
    for((rank,c) in ranked.withIndex()) {
     append("${rank+1}. ${c.label} • similarity ")
     append(String.format(Locale.US,"%.4f",c.similarity))
     append("\n")
    }
    append("\nA = old 4+5 baseline.")
    append("\nB keeps Profile 4 style/flow but 4+5 speaker timbre.")
    append("\nC/D gradually add Profile 5 style.")
    append("\nUse «Закрепить выбранный профиль» only after listening.")
   }

   return SearchResult(candidates,ranked,report)
  } catch(t:Throwable) {
   cacheDir.deleteRecursively()
   if(previousCache.exists()) previousCache.copyRecursively(cacheDir,overwrite=true)
   throw t
  }
 }

 fun activate(candidate:Candidate) {
  require(candidate.cacheSnapshot.exists()) {
   "Candidate cache missing: ${candidate.label}"
  }
  cacheDir.deleteRecursively()
  candidate.cacheSnapshot.copyRecursively(cacheDir,overwrite=true)

  val locked=File(filesDir,"conditioning_cache_3k_locked")
  locked.deleteRecursively()
  cacheDir.copyRecursively(locked,overwrite=true)
  File(locked,"winner.txt").writeText(
   "label=${candidate.label}\n"+
    "cond_profile4=${candidate.condWeight4}\n"+
    "cond_profile5=${candidate.condWeight5}\n"+
    "speaker_profile4=0.5\n"+
    "speaker_profile5=0.5\n"+
    "similarity=${String.format(Locale.US,"%.6f",candidate.similarity)}\n"
  )
 }

 private fun weightedAverage(
  vectors:List<FloatArray>,
  weights:FloatArray,
  expectedSize:Int
 ):FloatArray {
  require(vectors.isNotEmpty() && vectors.size==weights.size)
  val sumWeights=weights.sum()
  require(sumWeights>0f) { "Weights must sum to > 0" }
  val out=FloatArray(expectedSize)
  for(vIndex in vectors.indices) {
   val v=vectors[vIndex]
   require(v.size==expectedSize) { "Vector size ${v.size}, expected $expectedSize" }
   val w=weights[vIndex]/sumWeights
   for(i in out.indices) out[i]+=v[i]*w
  }
  return out
 }

 private fun readFloat32LE(file:File):FloatArray {
  require(file.exists()) { "Missing cache file: ${file.name}" }
  val bytes=file.readBytes()
  require(bytes.size%4==0) { "Invalid float32 file: ${file.name}" }
  val bb=ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
  return FloatArray(bytes.size/4) { bb.float }
 }

 private fun writeFloat32LE(file:File,data:FloatArray) {
  val bb=ByteBuffer.allocate(data.size*4).order(ByteOrder.LITTLE_ENDIAN)
  for(v in data) bb.putFloat(v)
  file.writeBytes(bb.array())
 }
}
''')

# ---------- MainActivity ----------
s = main.read_text()

assert 'text="XTTS-v2 Android V2 • Stage 3J"' in s, "Stage 3J title marker not found"
s = s.replace('text="XTTS-v2 Android V2 • Stage 3J"', 'text="XTTS-v2 Android V2 • Stage 3K"', 1)

assert 'text="Russian XTTS-v2 • multi-reference conditioning search"' in s, "Stage 3J subtitle marker not found"
s = s.replace(
    'text="Russian XTTS-v2 • multi-reference conditioning search"',
    'text="Russian XTTS-v2 • smooth 4+5 hybrid voice search"',
    1
)

assert 'append("Ready for Stage 3A / 3B / 3C / 3D / 3E / 3F / 3G / 3H / 3I / 3J.")' in s, "Ready marker not found"
s = s.replace(
    'append("Ready for Stage 3A / 3B / 3C / 3D / 3E / 3F / 3G / 3H / 3I / 3J.")',
    'append("Ready for Stage 3A / 3B / 3C / 3D / 3E / 3F / 3G / 3H / 3I / 3J / 3K.")',
    1
)

multi_btn = '  val multiReference=Button(this).apply { text="Stage 3J • Combine Profiles 3/4/5" }\n'
assert multi_btn in s, "Stage 3J button marker not found"
s = s.replace(
    multi_btn,
    multi_btn + '  val smoothHybrid=Button(this).apply { text="Stage 3K • Smooth the 4+5 voice" }\n',
    1
)

multi_add = '  content.addView(multiReference)\n'
assert multi_add in s, "Stage 3J addView marker not found"
s = s.replace(
    multi_add,
    multi_add + '  content.addView(smoothHybrid)\n',
    1
)

busy = '   multiReference.isEnabled=!busy\n'
assert busy in s, "Busy marker not found"
s = s.replace(
    busy,
    busy + '   smoothHybrid.isEnabled=!busy\n',
    1
)

play_marker = '  play.setOnClickListener {\n'
assert play_marker in s, "Play marker not found"

handler = r'''  smoothHybrid.setOnClickListener {
   val longRef=File(filesDir,"reference_long.wav")
   if(!longRef.exists()) {
    status.text="Stage 3K: reference_long.wav missing. Import the long Shorts WAV first."
    return@setOnClickListener
   }
   val typed=inputText.text?.toString().orEmpty()
   if(typed.isBlank()) {
    status.text="Stage 3K: enter Russian test text first."
    return@setOnClickListener
   }

   val preset=qualityMode.selectedItemPosition
   val fixedSeed=12346
   val generationDialog=GenerationProgressDialog(this,filesDir)
   generationDialog.show()
   setGenerationBusy(true)
   lastReport=""
   status.text="Stage 3K: testing smooth 4+5 hybrids…"

   Thread {
    val engine=XttsSmoothHybridSearch(filesDir)
    val result=runCatching {
     engine.run(
      longReference=longRef,
      text=typed,
      preset=preset,
      seed=fixedSeed
     ) { message ->
      runOnUiThread {
       status.text=message
       generationDialog.update(message)
      }
     }
    }

    runOnUiThread {
     setGenerationBusy(false)
     result.onSuccess { search ->
      lastReport=search.report
      val allNames=search.candidates.map { it.generatedFile.name }
      val first=search.ranked.first()
      refreshResults(allNames,first.generatedFile.name)
      copyReport.isEnabled=true
      play.isEnabled=true
      saveWav.isEnabled=true
      shareWav.isEnabled=true
      status.text=search.report

      generationDialog.complete(
       search.ranked.map { candidate ->
        GenerationProgressDialog.ResultItem(
         fileName=candidate.generatedFile.name,
         seed=fixedSeed,
         similarity=candidate.similarity,
         displayName=candidate.label
        )
       }
      ) { selected ->
       val chosen=search.candidates.firstOrNull {
        it.generatedFile.name==selected.fileName
       } ?: error("Selected Stage 3K candidate not found")

       engine.activate(chosen)
       refreshResults(allNames,chosen.generatedFile.name)
       lastReport=search.report+
        "\n\nLOCKED BY EAR: ${chosen.label}"+
        "\nFuture generations now use this Stage 3K conditioning."
       status.text=lastReport
       copyReport.isEnabled=true
      }
     }.onFailure {
      lastReport=""
      val error="Stage 3K failed: ${it.javaClass.simpleName}: ${it.message}"
      status.text=error
      generationDialog.fail(error)
     }
    }
   }.start()
  }

'''
s = s.replace(play_marker, handler + play_marker, 1)
main.write_text(s)

print("Stage 3K applied successfully")
print("Version: code 14 / 0.3.10-stage3k")
print("A: old 4+5 baseline (cond 50/50, speaker 50/50)")
print("B: cond 100% Profile 4, speaker 50/50 Profile 4+5")
print("C: cond 75% Profile 4 + 25% Profile 5, speaker 50/50")
print("D: cond 85% Profile 4 + 15% Profile 5, speaker 50/50")
print("Fixed comparison seed: 12346")
print("No automatic lock by similarity")
print("After listening, use 'Закрепить выбранный профиль'")
print("Previous Stage 3J conditioning remains active until you manually lock a Stage 3K candidate")
