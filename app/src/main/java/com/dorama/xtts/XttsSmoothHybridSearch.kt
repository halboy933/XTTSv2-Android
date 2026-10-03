package com.dorama.xtts

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
