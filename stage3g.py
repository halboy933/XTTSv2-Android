from pathlib import Path

root = Path(".")
gradle = root / "app/build.gradle.kts"
main = root / "app/src/main/java/com/dorama/xtts/MainActivity.kt"
conditioning = root / "app/src/main/java/com/dorama/xtts/XttsConditioning.kt"
pkg = root / "app/src/main/java/com/dorama/xtts"
dialog_file = pkg / "GenerationProgressDialog.kt"
similarity_file = pkg / "XttsVoiceSimilarity.kt"

for p in (gradle, main, conditioning):
    if not p.exists():
        raise SystemExit(f"Missing expected file: {p}")

# ---------- version ----------
s = gradle.read_text()
assert 'versionCode = 9' in s, "Expected Stage 3F versionCode 9"
assert 'versionName = "0.3.5-stage3f"' in s, "Expected Stage 3F versionName"
s = s.replace('versionCode = 9', 'versionCode = 10')
s = s.replace('versionName = "0.3.5-stage3f"', 'versionName = "0.3.6-stage3g"')
gradle.write_text(s)

# ---------- expose speaker embedding calculation from existing Stage 3A pipeline ----------
s = conditioning.read_text()
anchor = ''' data class TensorResult(val shape:LongArray,val data:FloatArray)

 fun run(referenceFile:File, progress:(String)->Unit):String {'''
insert = ''' data class TensorResult(val shape:LongArray,val data:FloatArray)

 fun computeSpeakerEmbedding(wavFile:File):FloatArray {
  require(wavFile.exists()) { "WAV not found: ${wavFile.name}" }
  val speakerModel=findRequired("speaker_encoder.onnx")
  val wav=readWav(wavFile)
  require(wav.samples.isNotEmpty()) { "WAV is empty" }
  val audio16=resampleSinc(wav.samples,wav.sampleRate,16000)
  val mel64=speakerMel(audio16)
  val speaker=runModel(
   speakerModel,
   "mel_spec",
   mel64.data,
   longArrayOf(1,mel64.channels.toLong(),mel64.frames.toLong())
  )
  require(speaker.shape.contentEquals(longArrayOf(1,512,1))) {
   "Unexpected speaker_embedding shape: ${shape(speaker.shape)}"
  }
  require(speaker.data.all { it.isFinite() }) { "speaker_embedding contains NaN/Inf" }
  return speaker.data
 }

 fun run(referenceFile:File, progress:(String)->Unit):String {'''
assert anchor in s, "XttsConditioning insertion anchor not found"
s = s.replace(anchor, insert, 1)
conditioning.write_text(s)

# ---------- voice similarity helper ----------
similarity_file.write_text(r'''package com.dorama.xtts

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

class XttsVoiceSimilarity(private val filesDir:File) {
 private val referenceEmbeddingFile=File(filesDir,"conditioning_cache/speaker_embedding.f32")

 fun score(generatedWav:File):Double {
  require(referenceEmbeddingFile.exists()) {
   "Reference speaker embedding missing. Run Stage 3A first."
  }
  val reference=readFloat32LE(referenceEmbeddingFile)
  require(reference.size==512) { "Reference speaker embedding invalid: ${reference.size}" }

  val generated=XttsConditioning(filesDir).computeSpeakerEmbedding(generatedWav)
  require(generated.size==512) { "Generated speaker embedding invalid: ${generated.size}" }

  return cosineSimilarity(reference,generated)
 }

 private fun cosineSimilarity(a:FloatArray,b:FloatArray):Double {
  require(a.size==b.size && a.isNotEmpty())
  var dot=0.0
  var aa=0.0
  var bb=0.0
  for(i in a.indices) {
   val x=a[i].toDouble()
   val y=b[i].toDouble()
   dot+=x*y
   aa+=x*x
   bb+=y*y
  }
  require(aa>0.0 && bb>0.0) { "Zero-length speaker embedding" }
  return dot/(sqrt(aa)*sqrt(bb))
 }

 private fun readFloat32LE(file:File):FloatArray {
  val bytes=file.readBytes()
  require(bytes.size%4==0) { "Invalid float32 file: ${file.name}" }
  val bb=ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
  return FloatArray(bytes.size/4) { bb.float }
 }
}
''')

# ---------- modal generation/progress/listen window ----------
dialog_file.write_text(r'''package com.dorama.xtts

import android.app.Activity
import android.app.Dialog
import android.media.MediaPlayer
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import java.io.File
import java.util.Locale

class GenerationProgressDialog(
 private val activity:Activity,
 private val filesDir:File
) {
 data class ResultItem(
  val fileName:String,
  val seed:Int,
  val similarity:Double
 )

 private val dialog=Dialog(activity)
 private val title=TextView(activity)
 private val progress=ProgressBar(activity)
 private val message=TextView(activity)
 private val resultLabel=TextView(activity)
 private val resultSpinner=Spinner(activity)
 private val playButton=Button(activity)
 private val closeButton=Button(activity)
 private var player:MediaPlayer?=null
 private var items:List<ResultItem> = emptyList()

 init {
  val content=LinearLayout(activity).apply {
   orientation=LinearLayout.VERTICAL
   setPadding(40,36,40,36)
  }
  val scroll=ScrollView(activity).apply {
   isFillViewport=true
   addView(content)
  }

  title.apply {
   text="Генерация голоса"
   textSize=24f
   gravity=Gravity.CENTER_HORIZONTAL
  }
  progress.apply {
   isIndeterminate=true
  }
  message.apply {
   text="Подготовка…"
   textSize=16f
   setPadding(0,20,0,20)
  }
  resultLabel.apply {
   text="Результаты"
   textSize=18f
   visibility=View.GONE
  }
  resultSpinner.visibility=View.GONE
  playButton.apply {
   text="▶ Прослушать выбранный вариант"
   visibility=View.GONE
  }
  closeButton.apply {
   text="Закрыть"
   isEnabled=false
  }

  content.addView(title)
  content.addView(progress)
  content.addView(message)
  content.addView(resultLabel)
  content.addView(resultSpinner)
  content.addView(playButton)
  content.addView(closeButton)

  dialog.setContentView(scroll)
  dialog.setCancelable(false)
  dialog.setCanceledOnTouchOutside(false)

  playButton.setOnClickListener {
   val pos=resultSpinner.selectedItemPosition
   if(pos !in items.indices) return@setOnClickListener
   val wav=File(filesDir,items[pos].fileName)
   if(!wav.exists()) {
    message.text="Файл не найден: ${wav.name}"
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
    message.text="Ошибка воспроизведения: ${it.message}"
   }
  }

  closeButton.setOnClickListener {
   player?.release()
   player=null
   dialog.dismiss()
  }

  dialog.setOnDismissListener {
   player?.release()
   player=null
  }
 }

 fun show() {
  dialog.show()
  dialog.window?.setLayout(
   ViewGroup.LayoutParams.MATCH_PARENT,
   ViewGroup.LayoutParams.WRAP_CONTENT
  )
 }

 fun update(text:String) {
  if(!dialog.isShowing) return
  message.text=text
 }

 fun complete(results:List<ResultItem>) {
  if(!dialog.isShowing) return
  items=results.sortedByDescending { it.similarity }
  title.text="✓ Генерация завершена"
  progress.visibility=View.GONE
  message.text=buildString {
   append("Готово. Выберите вариант и нажмите «Прослушать».\\n")
   if(items.isNotEmpty()) {
    val best=items.first()
    append("Лучший по speaker similarity: seed ${best.seed} • ")
    append(String.format(Locale.US,"%.4f",best.similarity))
   }
  }
  resultLabel.visibility=View.VISIBLE
  resultSpinner.visibility=View.VISIBLE
  playButton.visibility=View.VISIBLE
  closeButton.isEnabled=true

  val labels=items.mapIndexed { index,item ->
   val prefix=if(index==0) "★ " else ""
   prefix+"seed ${item.seed} • similarity "+String.format(Locale.US,"%.4f",item.similarity)
  }
  resultSpinner.adapter=ArrayAdapter(
   activity,
   android.R.layout.simple_spinner_dropdown_item,
   labels
  )
  resultSpinner.setSelection(0)
 }

 fun fail(text:String) {
  if(!dialog.isShowing) return
  title.text="Генерация остановлена"
  progress.visibility=View.GONE
  message.text=text
  closeButton.isEnabled=true
 }
}
''')

# ---------- MainActivity wiring ----------
s = main.read_text()

s = s.replace('text="XTTS-v2 Android V2 • Stage 3F"', 'text="XTTS-v2 Android V2 • Stage 3G"')
s = s.replace(
    'text="Russian XTTS-v2 • reproducible seeds + 3 variants"',
    'text="Russian XTTS-v2 • voice similarity + generation window"'
)
s = s.replace(
    'val synthesize=Button(this).apply { text="Generate 1 variant (Stage 3F)" }',
    'val synthesize=Button(this).apply { text="Generate 1 variant (Stage 3G)" }'
)
s = s.replace(
    'append("Ready for Stage 3A / 3B / 3C / 3D / 3E / 3F.")',
    'append("Ready for Stage 3A / 3B / 3C / 3D / 3E / 3F / 3G.")'
)

old = '''   val sampling=selectedSampling(seed)
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
   }.start()'''
new = '''   val sampling=selectedSampling(seed)
   val outputName="xtts_seed_${seed}.wav"
   val generationDialog=GenerationProgressDialog(this,filesDir)
   generationDialog.show()

   setGenerationBusy(true)
   lastReport=""
   status.text="Stage 3G: generating seed $seed…"

   Thread {
    val result=runCatching {
     val report=XttsSynthesisStage3C(filesDir).run(
      typed,
      sampling,
      outputName
     ) { message ->
      runOnUiThread {
       status.text="Seed $seed\\n$message"
       generationDialog.update("Seed $seed\\n$message")
      }
     }
     runOnUiThread { generationDialog.update("Оцениваю сходство голоса…") }
     val similarity=XttsVoiceSimilarity(filesDir).score(File(filesDir,outputName))
     Pair(report,similarity)
    }
    runOnUiThread {
     setGenerationBusy(false)
     result.onSuccess {
      val report=it.first
      val similarity=it.second
      lastReport=report+"\\nVoice similarity: "+String.format(java.util.Locale.US,"%.4f",similarity)
      refreshResults(listOf(outputName),outputName)
      copyReport.isEnabled=true
      play.isEnabled=true
      saveWav.isEnabled=true
      shareWav.isEnabled=true
      status.text=lastReport
      generationDialog.complete(
       listOf(GenerationProgressDialog.ResultItem(outputName,seed,similarity))
      )
     }.onFailure {
      lastReport=""
      val error="Stage 3G failed: ${it.javaClass.simpleName}: ${it.message}"
      status.text=error
      generationDialog.fail(error)
     }
    }
   }.start()'''
assert old in s, "Single generation block not found"
s = s.replace(old, new, 1)

old = '''   val preset=qualityMode.selectedItemPosition
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
   }.start()'''
new = '''   val preset=qualityMode.selectedItemPosition
   val seeds=intArrayOf(baseSeed,baseSeed+1,baseSeed+2)
   val names=seeds.map { "xtts_seed_${it}.wav" }
   val generationDialog=GenerationProgressDialog(this,filesDir)
   generationDialog.show()

   setGenerationBusy(true)
   lastReport=""
   status.text="Stage 3G: generating 3 variants…"

   Thread {
    val reports=ArrayList<String>()
    val similarities=DoubleArray(seeds.size) { Double.NaN }
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
      val report=XttsSynthesisStage3C(filesDir).run(
       typed,
       sampling,
       names[i]
      ) { message ->
       runOnUiThread {
        val progressText="Вариант ${i+1}/3 • seed $seed\\n$message"
        status.text=progressText
        generationDialog.update(progressText)
       }
      }
      runOnUiThread {
       generationDialog.update("Вариант ${i+1}/3 • seed $seed\\nОцениваю сходство голоса…")
      }
      val similarity=XttsVoiceSimilarity(filesDir).score(File(filesDir,names[i]))
      Pair(report,similarity)
     }
     if(result.isFailure) {
      failure=result.exceptionOrNull()
      break
     }
     val pair=result.getOrThrow()
     similarities[i]=pair.second
     reports.add(
      "VARIANT ${i+1}/3 • seed $seed\\n"+
       pair.first+
       "\\nVoice similarity: "+
       String.format(java.util.Locale.US,"%.4f",pair.second)
     )
    }

    runOnUiThread {
     setGenerationBusy(false)
     if(failure==null) {
      lastReport=reports.joinToString("\\n\\n====================\\n\\n")
      val bestIndex=similarities.indices.maxByOrNull { similarities[it] } ?: 0
      refreshResults(names,names[bestIndex])
      copyReport.isEnabled=true
      play.isEnabled=true
      saveWav.isEnabled=true
      shareWav.isEnabled=true
      status.text=buildString {
       append("STAGE 3G SUCCESS — 3 variants + voice similarity\\n\\n")
       for(i in seeds.indices) {
        append("${i+1}. seed ${seeds[i]} → ${names[i]} • similarity ")
        append(String.format(java.util.Locale.US,"%.4f",similarities[i]))
        append("\\n")
       }
       append("\\nBest speaker similarity: seed ${seeds[bestIndex]}")
      }
      generationDialog.complete(
       seeds.indices.map {
        GenerationProgressDialog.ResultItem(names[it],seeds[it],similarities[it])
       }
      )
     } else {
      lastReport=reports.joinToString("\\n\\n")
      val error="Stage 3G variant generation failed: ${failure?.javaClass?.simpleName}: ${failure?.message}"
      status.text=error
      generationDialog.fail(error)
     }
    }
   }.start()'''
assert old in s, "Three generation block not found"
s = s.replace(old, new, 1)

main.write_text(s)

print("Stage 3G applied successfully")
print("Version: code 10 / 0.3.6-stage3g")
print("Added modal generation progress window")
print("Completion window includes result list + playback")
print("Added speaker_encoder cosine similarity against cached reference voice")
print("Three variants are ranked by speaker similarity")
