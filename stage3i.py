from pathlib import Path

root = Path('.')
gradle = root / 'app/build.gradle.kts'
main = root / 'app/src/main/java/com/dorama/xtts/MainActivity.kt'
conditioning = root / 'app/src/main/java/com/dorama/xtts/XttsConditioning.kt'
similarity = root / 'app/src/main/java/com/dorama/xtts/XttsVoiceSimilarity.kt'
dialog = root / 'app/src/main/java/com/dorama/xtts/GenerationProgressDialog.kt'
profile_search = root / 'app/src/main/java/com/dorama/xtts/XttsVoiceProfileSearch.kt'

for p in (gradle, main, conditioning, similarity, dialog):
    if not p.exists():
        raise SystemExit(f'Missing expected file: {p}')

s = gradle.read_text()
assert 'versionCode = 11' in s, 'Expected Stage 3H versionCode 11'
assert 'versionName = "0.3.7-stage3h"' in s, 'Expected Stage 3H versionName'
s = s.replace('versionCode = 11', 'versionCode = 12')
s = s.replace('versionName = "0.3.7-stage3h"', 'versionName = "0.3.8-stage3i"')
gradle.write_text(s)

s = conditioning.read_text()
marker = ' data class TensorResult(val shape:LongArray,val data:FloatArray)\n'
assert marker in s, 'XttsConditioning TensorResult marker not found'
insert = marker + r'''
 data class ReferenceSegment(
  val file:File,
  val index:Int,
  val startSec:Double,
  val durationSec:Double
 )

 fun createReferenceSegments(
  sourceFile:File,
  segmentSeconds:Double=5.5,
  maxSegments:Int=5
 ):List<ReferenceSegment> {
  require(sourceFile.exists()) { "Long reference WAV not found" }
  require(maxSegments>=1) { "maxSegments must be >= 1" }
  val wav=readWav(sourceFile)
  val totalSec=wav.samples.size.toDouble()/wav.sampleRate
  require(totalSec>=10.0) {
   "Long Shorts WAV should contain at least 10 seconds of clean speech; got %.2f s".format(totalSec)
  }

  val segmentSamples=(segmentSeconds*wav.sampleRate)
   .roundToInt()
   .coerceAtMost(wav.samples.size)
  require(segmentSamples>=wav.sampleRate*3) { "Reference segment is too short" }

  val available=wav.samples.size-segmentSamples
  val count=if(available<=0) 1 else maxSegments
  val outDir=File(filesDir,"profile_segments")
  outDir.deleteRecursively()
  outDir.mkdirs()

  return (0 until count).map { i ->
   val start=if(count==1) 0 else ((available.toLong()*i)/(count-1)).toInt()
   val end=(start+segmentSamples).coerceAtMost(wav.samples.size)
   val samples=wav.samples.copyOfRange(start,end)
   val file=File(outDir,"reference_segment_${i+1}.wav")
   writePcm16Wav(file,samples,wav.sampleRate)
   ReferenceSegment(
    file=file,
    index=i+1,
    startSec=start.toDouble()/wav.sampleRate,
    durationSec=samples.size.toDouble()/wav.sampleRate
   )
  }
 }
'''
s = s.replace(marker, insert, 1)

read_wav_marker = ' private fun readWav(file:File):WavData {\n'
assert read_wav_marker in s, 'XttsConditioning readWav marker not found'
writer = r''' private fun writePcm16Wav(file:File,samples:FloatArray,sampleRate:Int) {
  val dataSize=samples.size*2
  val bb=ByteBuffer.allocate(44+dataSize).order(ByteOrder.LITTLE_ENDIAN)
  bb.put("RIFF".toByteArray(StandardCharsets.US_ASCII))
  bb.putInt(36+dataSize)
  bb.put("WAVE".toByteArray(StandardCharsets.US_ASCII))
  bb.put("fmt ".toByteArray(StandardCharsets.US_ASCII))
  bb.putInt(16)
  bb.putShort(1.toShort())
  bb.putShort(1.toShort())
  bb.putInt(sampleRate)
  bb.putInt(sampleRate*2)
  bb.putShort(2.toShort())
  bb.putShort(16.toShort())
  bb.put("data".toByteArray(StandardCharsets.US_ASCII))
  bb.putInt(dataSize)
  for(sample in samples) {
   val v=(sample.coerceIn(-1f,1f)*32767f)
    .roundToInt()
    .coerceIn(-32768,32767)
   bb.putShort(v.toShort())
  }
  file.writeBytes(bb.array())
 }

'''
s = s.replace(read_wav_marker, writer + read_wav_marker, 1)
conditioning.write_text(s)

s = similarity.read_text()
score_marker = ' fun score(generatedWav:File):Double {\n'
assert score_marker in s, 'XttsVoiceSimilarity score marker not found'
score_against = r''' fun scoreAgainstEmbedding(reference:FloatArray,generatedWav:File):Double {
  require(reference.size==512) { "Reference speaker embedding invalid: ${reference.size}" }
  val generated=XttsConditioning(filesDir).computeSpeakerEmbedding(generatedWav)
  require(generated.size==512) { "Generated speaker embedding invalid: ${generated.size}" }
  return cosineSimilarity(reference,generated)
 }

'''
s = s.replace(score_marker, score_against + score_marker, 1)
similarity.write_text(s)

s = dialog.read_text()
old_item = ''' data class ResultItem(
  val fileName:String,
  val seed:Int,
  val similarity:Double
 )
'''
new_item = ''' data class ResultItem(
  val fileName:String,
  val seed:Int,
  val similarity:Double,
  val displayName:String?=null
 )
'''
assert old_item in s, 'GenerationProgressDialog ResultItem marker not found'
s = s.replace(old_item, new_item, 1)
old_best = '''    append("Лучший по speaker similarity: seed ${best.seed} • ")
    append(String.format(Locale.US,"%.4f",best.similarity))
'''
new_best = '''    append("Лучший по speaker similarity: ")
    append(best.displayName ?: "seed ${best.seed}")
    append(" • ")
    append(String.format(Locale.US,"%.4f",best.similarity))
'''
assert old_best in s, 'GenerationProgressDialog best label marker not found'
s = s.replace(old_best, new_best, 1)
old_labels = '''  val labels=items.mapIndexed { index,item ->
   val prefix=if(index==0) "★ " else ""
   prefix+"seed ${item.seed} • similarity "+String.format(Locale.US,"%.4f",item.similarity)
  }
'''
new_labels = '''  val labels=items.mapIndexed { index,item ->
   val prefix=if(index==0) "★ " else ""
   val name=item.displayName ?: "seed ${item.seed}"
   prefix+name+" • similarity "+String.format(Locale.US,"%.4f",item.similarity)
  }
'''
assert old_labels in s, 'GenerationProgressDialog labels marker not found'
s = s.replace(old_labels, new_labels, 1)
dialog.write_text(s)

profile_search.write_text(r'''package com.dorama.xtts

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
''')

s = main.read_text()
s = s.replace('text="XTTS-v2 Android V2 • Stage 3H"', 'text="XTTS-v2 Android V2 • Stage 3I"', 1)
s = s.replace('text="Russian XTTS-v2 • 8-variant voice search + Top 3"', 'text="Russian XTTS-v2 • reference-profile search + locked voice"', 1)
s = s.replace('append("Ready for Stage 3A / 3B / 3C / 3D / 3E / 3F / 3G / 3H.")', 'append("Ready for Stage 3A / 3B / 3C / 3D / 3E / 3F / 3G / 3H / 3I.")', 1)

button_marker = '  val import=Button(this).apply { text="Import voice WAV (3–6 sec)" }\n'
assert button_marker in s, 'MainActivity import button marker not found'
s = s.replace(button_marker, button_marker + '  val importLong=Button(this).apply { text="Import long Shorts WAV (15+ sec)" }\n', 1)
search_button_marker = '  val synthesize3=Button(this).apply { text="Find best voice • 8 variants → Top 3" }\n'
assert search_button_marker in s, 'MainActivity Stage 3H button marker not found'
s = s.replace(search_button_marker, search_button_marker + '  val profileSearch=Button(this).apply { text="Stage 3I • Find best reference profile" }\n', 1)
content_import_marker = '  content.addView(import)\n'
assert content_import_marker in s, 'MainActivity import addView marker not found'
s = s.replace(content_import_marker, content_import_marker + '  content.addView(importLong)\n', 1)
content_search_marker = '  content.addView(synthesize3)\n'
assert content_search_marker in s, 'MainActivity synthesize3 addView marker not found'
s = s.replace(content_search_marker, content_search_marker + '  content.addView(profileSearch)\n', 1)

download_listener_marker = '  download.setOnClickListener {\n'
assert download_listener_marker in s, 'MainActivity download listener marker not found'
long_listener = '''  importLong.setOnClickListener {
   startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
    type="audio/*"
    addCategory(Intent.CATEGORY_OPENABLE)
   },10)
  }

'''
s = s.replace(download_listener_marker, long_listener + download_listener_marker, 1)

busy_marker = '   resultMode.isEnabled=!busy\n'
assert busy_marker in s, 'MainActivity busy marker not found'
s = s.replace(busy_marker, busy_marker + '   importLong.isEnabled=!busy\n   profileSearch.isEnabled=!busy\n', 1)

play_marker = '  play.setOnClickListener {\n'
assert play_marker in s, 'MainActivity play marker not found'
search_handler = r'''  profileSearch.setOnClickListener {
   val longRef=File(filesDir,"reference_long.wav")
   if(!longRef.exists()) {
    status.text="Stage 3I: import a long Shorts WAV first."
    return@setOnClickListener
   }
   val typed=inputText.text?.toString().orEmpty()
   if(typed.isBlank()) {
    status.text="Stage 3I: enter Russian test text first."
    return@setOnClickListener
   }

   val preset=qualityMode.selectedItemPosition
   val fixedSeed=12346
   val generationDialog=GenerationProgressDialog(this,filesDir)
   generationDialog.show()
   setGenerationBusy(true)
   lastReport=""
   status.text="Stage 3I: searching for the best reference profile…"

   Thread {
    val result=runCatching {
     XttsVoiceProfileSearch(filesDir).run(
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
         displayName="Профиль ${candidate.profileIndex} • ${String.format(java.util.Locale.US,"%.1f",candidate.startSec)} сек"
        )
       }
      )
     }.onFailure {
      lastReport=""
      val error="Stage 3I failed: ${it.javaClass.simpleName}: ${it.message}"
      status.text=error
      generationDialog.fail(error)
     }
    }
   }.start()
  }

'''
s = s.replace(play_marker, search_handler + play_marker, 1)

activity_result_marker = '  if(requestCode==9) {\n'
assert activity_result_marker in s, 'MainActivity onActivityResult marker not found'
long_result = '''  if(requestCode==10) {
   val f=File(filesDir,"reference_long.wav")
   runCatching {
    contentResolver.openInputStream(u)!!.use { input ->
     f.outputStream().use { output -> input.copyTo(output) }
    }
   }.onSuccess {
    lastReport=""
    status.text="Long Shorts WAV imported: ${f.length()/1024} KB\\nReady for Stage 3I profile search."
   }.onFailure {
    status.text="Long Shorts WAV import failed: ${it.message}"
   }
   return
  }

'''
s = s.replace(activity_result_marker, long_result + activity_result_marker, 1)
main.write_text(s)

print('Stage 3I applied successfully')
print('Version: code 12 / 0.3.8-stage3i')
print('Long Shorts WAV import added')
print('5 x 5.5 s reference profiles sampled across the recording')
print('Stable speaker target = average embedding across all 5 profiles')
print('Each profile uses the same seed 12346 for fair comparison')
print('Top 3 generated voices are shown for playback')
print('Best profile is automatically locked as reference.wav')
print('Previous working reference/cache are restored automatically if search fails')
