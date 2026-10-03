package com.dorama.xtts

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
