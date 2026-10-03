package com.dorama.xtts

import java.io.File
import java.util.Locale

class XttsVoiceProfileSearch(private val filesDir:File) {
 data class Candidate(
  val profileIndex:Int,
  val startSec:Double,
  val segmentFile:File,
  val generatedFile:File,
  val similarity:Double
 )

 data class SearchResult(
  val candidates:List<Candidate>,
  val top3:List<Candidate>,
  val best:Candidate,
  val report:String
 )

 fun run(
  longReference:File,
  text:String,
  preset:Int,
  seed:Int=12346,
  progress:(String)->Unit
 ):SearchResult {
  require(longReference.exists()) { "Import a long Shorts WAV first." }
  require(text.isNotBlank()) { "Enter Russian test text first." }

  val conditioning=XttsConditioning(filesDir)
  val similarityEngine=XttsVoiceSimilarity(filesDir)
  progress("Stage 3I 1/4: preparing 5 reference profiles…")
  val segments=conditioning.createReferenceSegments(longReference,5.5,5)
  require(segments.isNotEmpty()) { "No reference segments were created." }

  progress("Stage 3I 2/4: building long-recording speaker target…")
  val embeddings=segments.mapIndexed { i,segment ->
   progress("Stage 3I 2/4: speaker target ${i+1}/${segments.size}…")
   conditioning.computeSpeakerEmbedding(segment.file)
  }
  val target=FloatArray(512)
  for(embedding in embeddings) {
   require(embedding.size==512) { "Unexpected speaker embedding size: ${embedding.size}" }
   for(i in target.indices) target[i]+=embedding[i]
  }
  for(i in target.indices) target[i]/=embeddings.size.toFloat()

  val cacheDir=File(filesDir,"conditioning_cache")
  val referenceFile=File(filesDir,"reference.wav")
  val backupDir=File(filesDir,"profile_search_backup")
  backupDir.deleteRecursively()
  backupDir.mkdirs()
  if(cacheDir.exists()) {
   cacheDir.copyRecursively(File(backupDir,"conditioning_cache"),overwrite=true)
  }
  if(referenceFile.exists()) {
   referenceFile.copyTo(File(backupDir,"reference.wav"),overwrite=true)
  }

  try {
   val candidates=ArrayList<Candidate>()
   for((i,segment) in segments.withIndex()) {
    progress(
     "Stage 3I 3/4: профиль ${i+1}/${segments.size}\n"+
      "Фрагмент %.1f–%.1f сек\nСоздаю conditioning…".format(
       Locale.US,
       segment.startSec,
       segment.startSec+segment.durationSec
      )
    )

    conditioning.run(segment.file) { message ->
     progress("Профиль ${i+1}/${segments.size}\n$message")
    }

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

    val generated=File(filesDir,"xtts_profile_${i+1}_seed_${seed}.wav")
    XttsSynthesisStage3C(filesDir).run(
     text,
     sampling,
     generated.name
    ) { message ->
     progress("Профиль ${i+1}/${segments.size} • seed $seed\n$message")
    }

    progress("Профиль ${i+1}/${segments.size}: оцениваю сходство…")
    val score=similarityEngine.scoreAgainstEmbedding(target,generated)
    candidates.add(
     Candidate(
      profileIndex=i+1,
      startSec=segment.startSec,
      segmentFile=segment.file,
      generatedFile=generated,
      similarity=score
     )
    )
   }

   val ranked=candidates.sortedByDescending { it.similarity }
   val best=ranked.first()
   val top3=ranked.take(3)

   progress("Stage 3I 4/4: закрепляю лучший профиль ${best.profileIndex}…")
   val bestSaved=File(filesDir,"reference_best_3i.wav")
   best.segmentFile.copyTo(bestSaved,overwrite=true)
   best.segmentFile.copyTo(referenceFile,overwrite=true)
   conditioning.run(referenceFile) { message ->
    progress("Stage 3I 4/4: $message")
   }

   val report=buildString {
    append("STAGE 3I SUCCESS — best reference profile locked\n\n")
    append("Long Shorts WAV: ${longReference.name}\n")
    append("Profiles tested: ${candidates.size}\n")
    append("Fixed test seed: $seed\n\n")
    for((rank,c) in ranked.withIndex()) {
     append("${rank+1}. Profile ${c.profileIndex} • start ")
     append(String.format(Locale.US,"%.1f s",c.startSec))
     append(" • similarity ")
     append(String.format(Locale.US,"%.4f",c.similarity))
     if(rank==0) append("  ★ LOCKED")
     append("\n")
    }
    append("\nBest profile ${best.profileIndex} saved as reference.wav")
    append("\nBackup copy: reference_best_3i.wav")
    append("\nFuture generation now uses this conditioning profile.")
   }

   backupDir.deleteRecursively()
   return SearchResult(candidates,top3,best,report)
  } catch(t:Throwable) {
   cacheDir.deleteRecursively()
   val cacheBackup=File(backupDir,"conditioning_cache")
   if(cacheBackup.exists()) cacheBackup.copyRecursively(cacheDir,overwrite=true)

   referenceFile.delete()
   val refBackup=File(backupDir,"reference.wav")
   if(refBackup.exists()) refBackup.copyTo(referenceFile,overwrite=true)

   backupDir.deleteRecursively()
   throw t
  }
 }
}
