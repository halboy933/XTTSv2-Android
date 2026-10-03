from pathlib import Path

root = Path(".")
gradle = root / "app/build.gradle.kts"
main = root / "app/src/main/java/com/dorama/xtts/MainActivity.kt"
multi = root / "app/src/main/java/com/dorama/xtts/XttsMultiReferenceSearch.kt"

for p in (gradle, main):
    if not p.exists():
        raise SystemExit(f"Missing expected file: {p}")

s = gradle.read_text()
assert 'versionCode = 12' in s, "Expected Stage 3I versionCode 12"
assert 'versionName = "0.3.8-stage3i"' in s, "Expected Stage 3I versionName"
s = s.replace('versionCode = 12', 'versionCode = 13')
s = s.replace('versionName = "0.3.8-stage3i"', 'versionName = "0.3.9-stage3j"')
gradle.write_text(s)

multi.write_text(r'''package com.dorama.xtts

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale

class XttsMultiReferenceSearch(private val filesDir:File) {
 data class Candidate(
  val id:String,
  val label:String,
  val profiles:IntArray,
  val generatedFile:File,
  val similarity:Double,
  val cacheSnapshot:File
 )

 data class SearchResult(
  val candidates:List<Candidate>,
  val top3:List<Candidate>,
  val best:Candidate,
  val report:String
 )

 private val cacheDir=File(filesDir,"conditioning_cache")

 fun run(
  longReference:File,
  text:String,
  preset:Int,
  seed:Int=12346,
  progress:(String)->Unit
 ):SearchResult {
  require(longReference.exists()) { "reference_long.wav is missing. Run/import Stage 3I material first." }
  require(text.isNotBlank()) { "Enter Russian test text first." }

  val conditioning=XttsConditioning(filesDir)
  val similarityEngine=XttsVoiceSimilarity(filesDir)

  progress("Stage 3J 1/5: preparing the same 5 reference profiles…")
  val segments=conditioning.createReferenceSegments(longReference,5.5,5)
  require(segments.size>=5) { "Stage 3J requires 5 reference profiles." }

  progress("Stage 3J 2/5: building stable speaker target…")
  val allEmbeddings=segments.mapIndexed { i,segment ->
   progress("Stage 3J 2/5: speaker target ${i+1}/5…")
   conditioning.computeSpeakerEmbedding(segment.file)
  }
  val target=average(allEmbeddings,512)

  val combinations=listOf(
   Triple("p4","Profile 4",intArrayOf(4)),
   Triple("p34","Profiles 3+4",intArrayOf(3,4)),
   Triple("p45","Profiles 4+5",intArrayOf(4,5)),
   Triple("p345","Profiles 3+4+5",intArrayOf(3,4,5))
  )

  val backupDir=File(filesDir,"multi_reference_backup")
  val candidatesRoot=File(filesDir,"multi_reference_candidates")
  backupDir.deleteRecursively()
  candidatesRoot.deleteRecursively()
  backupDir.mkdirs()
  candidatesRoot.mkdirs()

  if(cacheDir.exists()) {
   cacheDir.copyRecursively(File(backupDir,"conditioning_cache"),overwrite=true)
  }

  try {
   val candidates=ArrayList<Candidate>()

   for((comboIndex,combo) in combinations.withIndex()) {
    val id=combo.first
    val label=combo.second
    val profiles=combo.third

    progress(
     "Stage 3J 3/5: комбинация ${comboIndex+1}/${combinations.size}\n"+
      "$label\nСобираю multi-reference conditioning…"
    )

    val condVectors=ArrayList<FloatArray>()
    val speakerVectors=ArrayList<FloatArray>()

    for((partIndex,profileIndex) in profiles.withIndex()) {
     val segment=segments[profileIndex-1]
     conditioning.run(segment.file) { message ->
      progress(
       "Stage 3J 3/5: $label\n"+
        "Эталон ${partIndex+1}/${profiles.size} • Profile $profileIndex\n$message"
      )
     }

     val cond=readFloat32LE(File(cacheDir,"cond_latents.f32"))
     val speaker=readFloat32LE(File(cacheDir,"speaker_embedding.f32"))
     require(cond.size==32*1024) { "cond_latents invalid for Profile $profileIndex: ${cond.size}" }
     require(speaker.size==512) { "speaker_embedding invalid for Profile $profileIndex: ${speaker.size}" }
     condVectors.add(cond)
     speakerVectors.add(speaker)
    }

    val avgCond=average(condVectors,32*1024)
    val avgSpeaker=average(speakerVectors,512)

    cacheDir.mkdirs()
    writeFloat32LE(File(cacheDir,"cond_latents.f32"),avgCond)
    writeFloat32LE(File(cacheDir,"speaker_embedding.f32"),avgSpeaker)
    File(cacheDir,"meta.txt").writeText(
     "stage=3J\n"+
      "multi_reference=${profiles.joinToString("+")}\n"+
      "seed=$seed\n"
    )

    val snapshot=File(candidatesRoot,id)
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

    val generated=File(filesDir,"xtts_multi_${id}_seed_${seed}.wav")
    XttsSynthesisStage3C(filesDir).run(
     text,
     sampling,
     generated.name
    ) { message ->
     progress("Stage 3J 3/5: $label • seed $seed\n$message")
    }

    progress("Stage 3J 3/5: $label\nОцениваю сходство голоса…")
    val score=similarityEngine.scoreAgainstEmbedding(target,generated)

    candidates.add(
     Candidate(
      id=id,
      label=label,
      profiles=profiles,
      generatedFile=generated,
      similarity=score,
      cacheSnapshot=File(snapshot,"conditioning_cache")
     )
    )
   }

   progress("Stage 3J 4/5: ranking multi-reference profiles…")
   val ranked=candidates.sortedByDescending { it.similarity }
   val best=ranked.first()
   val top3=ranked.take(3)

   progress("Stage 3J 5/5: locking ${best.label}…")
   cacheDir.deleteRecursively()
   best.cacheSnapshot.copyRecursively(cacheDir,overwrite=true)

   val locked=File(filesDir,"conditioning_cache_3j_locked")
   locked.deleteRecursively()
   cacheDir.copyRecursively(locked,overwrite=true)
   File(locked,"winner.txt").writeText(
    "label=${best.label}\n"+
     "profiles=${best.profiles.joinToString("+")}\n"+
     "similarity=${String.format(Locale.US,"%.6f",best.similarity)}\n"+
     "seed=$seed\n"
   )

   val report=buildString {
    append("STAGE 3J SUCCESS — multi-reference conditioning\n\n")
    append("Fixed test seed: $seed\n")
    append("Combinations tested: ${candidates.size}\n\n")
    for((rank,c) in ranked.withIndex()) {
     append("${rank+1}. ${c.label} • similarity ")
     append(String.format(Locale.US,"%.4f",c.similarity))
     if(rank==0) append("  ★ LOCKED")
     append("\n")
    }
    append("\nBest: ${best.label}")
    append("\nActive conditioning cache now uses this multi-reference profile.")
    append("\nPersistent snapshot: conditioning_cache_3j_locked/")
    append("\nreference.wav itself was not replaced; Stage 3J works through the locked cache.")
   }

   backupDir.deleteRecursively()
   return SearchResult(candidates,top3,best,report)
  } catch(t:Throwable) {
   cacheDir.deleteRecursively()
   val previous=File(backupDir,"conditioning_cache")
   if(previous.exists()) previous.copyRecursively(cacheDir,overwrite=true)
   backupDir.deleteRecursively()
   throw t
  }
 }

 private fun average(vectors:List<FloatArray>,expectedSize:Int):FloatArray {
  require(vectors.isNotEmpty()) { "No vectors to average" }
  val out=FloatArray(expectedSize)
  for(v in vectors) {
   require(v.size==expectedSize) { "Vector size ${v.size}, expected $expectedSize" }
   for(i in out.indices) out[i]+=v[i]
  }
  val n=vectors.size.toFloat()
  for(i in out.indices) out[i]/=n
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

s = main.read_text()

assert 'text="XTTS-v2 Android V2 • Stage 3I"' in s, "Stage 3I title marker not found"
s = s.replace('text="XTTS-v2 Android V2 • Stage 3I"', 'text="XTTS-v2 Android V2 • Stage 3J"', 1)

assert 'text="Russian XTTS-v2 • reference-profile search + locked voice"' in s, "Stage 3I subtitle marker not found"
s = s.replace(
    'text="Russian XTTS-v2 • reference-profile search + locked voice"',
    'text="Russian XTTS-v2 • multi-reference conditioning search"',
    1
)

assert 'append("Ready for Stage 3A / 3B / 3C / 3D / 3E / 3F / 3G / 3H / 3I.")' in s, "Ready marker not found"
s = s.replace(
    'append("Ready for Stage 3A / 3B / 3C / 3D / 3E / 3F / 3G / 3H / 3I.")',
    'append("Ready for Stage 3A / 3B / 3C / 3D / 3E / 3F / 3G / 3H / 3I / 3J.")',
    1
)

profile_btn = '  val profileSearch=Button(this).apply { text="Stage 3I • Find best reference profile" }\n'
assert profile_btn in s, "Stage 3I button marker not found"
s = s.replace(
    profile_btn,
    profile_btn + '  val multiReference=Button(this).apply { text="Stage 3J • Combine Profiles 3/4/5" }\n',
    1
)

profile_add = '  content.addView(profileSearch)\n'
assert profile_add in s, "Stage 3I addView marker not found"
s = s.replace(profile_add, profile_add + '  content.addView(multiReference)\n', 1)

busy = '   profileSearch.isEnabled=!busy\n'
assert busy in s, "Busy marker not found"
s = s.replace(busy, busy + '   multiReference.isEnabled=!busy\n', 1)

play_marker = '  play.setOnClickListener {\n'
assert play_marker in s, "Play marker not found"

handler = r'''  multiReference.setOnClickListener {
   val longRef=File(filesDir,"reference_long.wav")
   if(!longRef.exists()) {
    status.text="Stage 3J: reference_long.wav missing. Import/run Stage 3I material first."
    return@setOnClickListener
   }
   val typed=inputText.text?.toString().orEmpty()
   if(typed.isBlank()) {
    status.text="Stage 3J: enter Russian test text first."
    return@setOnClickListener
   }

   val preset=qualityMode.selectedItemPosition
   val fixedSeed=12346
   val generationDialog=GenerationProgressDialog(this,filesDir)
   generationDialog.show()
   setGenerationBusy(true)
   lastReport=""
   status.text="Stage 3J: testing multi-reference conditioning…"

   Thread {
    val result=runCatching {
     XttsMultiReferenceSearch(filesDir).run(
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
      refreshResults(allNames,search.best.generatedFile.name)
      copyReport.isEnabled=true
      play.isEnabled=true
      saveWav.isEnabled=true
      shareWav.isEnabled=true
      status.text=search.report

      generationDialog.complete(
       search.top3.map { candidate ->
        GenerationProgressDialog.ResultItem(
         fileName=candidate.generatedFile.name,
         seed=fixedSeed,
         similarity=candidate.similarity,
         displayName=candidate.label
        )
       }
      )
     }.onFailure {
      lastReport=""
      val error="Stage 3J failed: ${it.javaClass.simpleName}: ${it.message}"
      status.text=error
      generationDialog.fail(error)
     }
    }
   }.start()
  }

'''
s = s.replace(play_marker, handler + play_marker, 1)
main.write_text(s)

print("Stage 3J applied successfully")
print("Version: code 13 / 0.3.9-stage3j")
print("Test combinations: Profile 4, Profiles 3+4, Profiles 4+5, Profiles 3+4+5")
print("Fixed comparison seed: 12346")
print("Each combination averages cond_latents + speaker_embedding")
print("Top 3 are shown for playback")
print("Best multi-reference conditioning cache is locked automatically")
print("Previous Stage 3I cache is restored automatically if Stage 3J fails")
