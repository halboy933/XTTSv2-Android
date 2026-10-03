package com.dorama.xtts

import android.app.Activity
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.media.MediaPlayer
import android.os.Bundle
import android.text.InputType
import android.view.Gravity
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.util.zip.ZipInputStream

class MainActivity : AppCompatActivity() {
 private lateinit var status: TextView
 private var refPath: String? = null
 private var lastReport: String = ""
 private var player: MediaPlayer? = null
 private var pendingSaveName: String? = null
 private val generatedNames=ArrayList<String>()
 private lateinit var resultMode: Spinner
 private val modelDir by lazy { File(filesDir, "xtts_models") }

 override fun onCreate(savedInstanceState: Bundle?) {
  super.onCreate(savedInstanceState)

  val content=LinearLayout(this).apply {
   orientation=LinearLayout.VERTICAL
   setPadding(32,32,32,32)
  }
  val scroll=ScrollView(this).apply {
   isFillViewport=true
   addView(content)
  }

  content.addView(TextView(this).apply {
   text="XTTS-v2 Android V2 • Stage 3M"
   textSize=24f
  })
  content.addView(TextView(this).apply {
   text="Russian XTTS-v2 • clean UI • Reference Lab + generation"
  })

  val import=Button(this).apply { text="Import voice WAV (3–6 sec)" }
  val importLong=Button(this).apply { text="Import long Shorts WAV (15+ sec)" }
  val inputText=EditText(this).apply {
   hint="Русский текст"
   setText("Сегодня я хочу рассказать вам об одной удивительной истории.")
   minLines=4
   gravity=Gravity.TOP or Gravity.START
   inputType=InputType.TYPE_CLASS_TEXT or
    InputType.TYPE_TEXT_FLAG_MULTI_LINE or
    InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
   setHorizontallyScrolling(false)
  }
  val qualityLabel=TextView(this).apply {
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
  val download=Button(this).apply { text="Import model ZIP / verify ONNX" }
  val conditioning=Button(this).apply { text="Compute voice conditioning (Stage 3A)" }
  val gptTest=Button(this).apply { text="Test Russian GPT (Stage 3B)" }
  val synthesize=Button(this).apply { text="Сгенерировать короткий текст" }
  val synthesize3=Button(this).apply { text="Find best voice • 8 variants → Top 3" }
  val profileSearch=Button(this).apply { text="Stage 3I • Find best reference profile" }
  val multiReference=Button(this).apply { text="Stage 3J • Combine Profiles 3/4/5" }
  val smoothHybrid=Button(this).apply { text="Stage 3K • Smooth the 4+5 voice" }
  val longSynthesis=Button(this).apply { text="Сгенерировать длинный текст → один WAV" }
  val referenceLab=Button(this).apply { text="Reference Lab • Найти лучшие эталоны голоса" }
  val play=Button(this).apply {
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
  val copyReport=Button(this).apply {
   text="Copy report"
   isEnabled=false
  }
  status=TextView(this).apply { setTextIsSelectable(true) }

  content.addView(import)
  content.addView(importLong)
  content.addView(inputText)
  content.addView(qualityLabel)
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
  content.addView(profileSearch)
  content.addView(multiReference)
  content.addView(smoothHybrid)
  content.addView(longSynthesis)
  content.addView(referenceLab)
  content.addView(play)
  content.addView(saveWav)
  content.addView(shareWav)
  content.addView(copyReport)
  listOf(
   import,
   download,
   conditioning,
   gptTest,
   synthesize3,
   profileSearch,
   multiReference,
   smoothHybrid
  ).forEach { it.visibility=android.view.View.GONE }
  content.addView(status)
  setContentView(scroll)

  val existingRef=File(filesDir,"reference.wav")
  if(existingRef.exists()) refPath=existingRef.absolutePath

  runCatching { OrtEnvironment.getEnvironment() }
   .onSuccess {
    val modelCount=modelDir.walkTopDown().count { it.isFile && it.extension.equals("onnx",true) }
    val cache=File(filesDir,"conditioning_cache/cond_latents.f32")
    status.text=buildString {
     append("Status: ONNX Runtime OK\n")
     append("Reference: ")
     append(if(existingRef.exists()) "found (${existingRef.length()/1024} KB)" else "not imported")
     append("\nONNX models: $modelCount found\n")
     append("Conditioning cache: ${if(cache.exists()) "found" else "missing"}\n")
     append("Generated WAV: ${if(File(filesDir,"xtts_generated.wav").exists()) "found" else "missing"}\n")
     append("Ready: short generation / long generation / Reference Lab.")
    }
   }
   .onFailure { status.text="ONNX Runtime error: ${it.message}" }

  copyReport.setOnClickListener {
   if(lastReport.isNotBlank()) {
    val clipboard=getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
    clipboard.setPrimaryClip(ClipData.newPlainText("XTTS report",lastReport))
    Toast.makeText(this,"Report copied",Toast.LENGTH_SHORT).show()
   }
  }

  import.setOnClickListener {
   startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
    type="audio/*"
    addCategory(Intent.CATEGORY_OPENABLE)
   },7)
  }

  importLong.setOnClickListener {
   startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
    type="audio/*"
    addCategory(Intent.CATEGORY_OPENABLE)
   },10)
  }

  download.setOnClickListener {
   startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
    type="application/zip"
    addCategory(Intent.CATEGORY_OPENABLE)
   },8)
  }

  conditioning.setOnClickListener {
   val ref=File(filesDir,"reference.wav")
   if(!ref.exists()) {
    status.text="Stage 3A: import reference WAV first."
    return@setOnClickListener
   }
   conditioning.isEnabled=false
   gptTest.isEnabled=false
   synthesize.isEnabled=false
   synthesize3.isEnabled=false
   copyReport.isEnabled=false
   lastReport=""
   status.text="Stage 3A: starting…"

   Thread {
    val result=runCatching {
     XttsConditioning(filesDir).run(ref) { message ->
      runOnUiThread { status.text=message }
     }
    }
    runOnUiThread {
     conditioning.isEnabled=true
     gptTest.isEnabled=true
     synthesize.isEnabled=true
     synthesize3.isEnabled=true
     result.onSuccess {
      lastReport=it
      copyReport.isEnabled=true
      status.text=it
     }.onFailure {
      lastReport=""
      copyReport.isEnabled=false
      status.text="Stage 3A failed: ${it.javaClass.simpleName}: ${it.message}"
     }
    }
   }.start()
  }

  gptTest.setOnClickListener {
   val typed=inputText.text?.toString().orEmpty()
   if(typed.isBlank()) {
    status.text="Stage 3B: enter Russian text first."
    return@setOnClickListener
   }
   conditioning.isEnabled=false
   gptTest.isEnabled=false
   synthesize.isEnabled=false
   synthesize3.isEnabled=false
   copyReport.isEnabled=false
   lastReport=""
   status.text="Stage 3B: starting…"

   Thread {
    val result=runCatching {
     XttsGptStage3B(filesDir).run(typed) { message ->
      runOnUiThread { status.text=message }
     }
    }
    runOnUiThread {
     conditioning.isEnabled=true
     gptTest.isEnabled=true
     synthesize.isEnabled=true
     synthesize3.isEnabled=true
     result.onSuccess {
      lastReport=it
      copyReport.isEnabled=true
      status.text=it
     }.onFailure {
      lastReport=""
      copyReport.isEnabled=false
      status.text="Stage 3B failed: ${it.javaClass.simpleName}: ${it.message}"
     }
    }
   }.start()
  }

  fun selectedSampling(seed:Int):XttsSynthesisStage3C.Sampling {
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
   importLong.isEnabled=!busy
   profileSearch.isEnabled=!busy
   multiReference.isEnabled=!busy
   smoothHybrid.isEnabled=!busy
   longSynthesis.isEnabled=!busy
   referenceLab.isEnabled=!busy
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
    status.text="Stage 3H: enter Russian text first."
    return@setOnClickListener
   }
   val seed=seedInput.text?.toString()?.trim()?.toIntOrNull()
   if(seed==null) {
    status.text="Stage 3H: seed must be a whole number."
    return@setOnClickListener
   }
   val sampling=selectedSampling(seed)
   val outputName="xtts_seed_${seed}.wav"
   val generationDialog=GenerationProgressDialog(this,filesDir)
   generationDialog.show()

   setGenerationBusy(true)
   lastReport=""
   status.text="Stage 3H: generating seed $seed…"

   Thread {
    val result=runCatching {
     val report=XttsSynthesisStage3C(filesDir).run(
      typed,
      sampling,
      outputName
     ) { message ->
      runOnUiThread {
       status.text="Seed $seed\n$message"
       generationDialog.update("Seed $seed\n$message")
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
      lastReport=report+"\nVoice similarity: "+String.format(java.util.Locale.US,"%.4f",similarity)
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
      val error="Stage 3H failed: ${it.javaClass.simpleName}: ${it.message}"
      status.text=error
      generationDialog.fail(error)
     }
    }
   }.start()
  }

  synthesize3.setOnClickListener {
   val typed=inputText.text?.toString().orEmpty()
   if(typed.isBlank()) {
    status.text="Stage 3H: enter Russian text first."
    return@setOnClickListener
   }
   val baseSeed=seedInput.text?.toString()?.trim()?.toIntOrNull()
   if(baseSeed==null || baseSeed>Int.MAX_VALUE-7) {
    status.text="Stage 3H: enter a valid seed (max ${Int.MAX_VALUE-7})."
    return@setOnClickListener
   }

   val preset=qualityMode.selectedItemPosition
   val seeds=IntArray(8) { baseSeed+it }
   val names=seeds.map { "xtts_seed_${it}.wav" }
   val generationDialog=GenerationProgressDialog(this,filesDir)
   generationDialog.show()

   setGenerationBusy(true)
   lastReport=""
   status.text="Stage 3H: searching 8 voice variants…"

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
        val progressText="Вариант ${i+1}/8 • seed $seed\n$message"
        status.text=progressText
        generationDialog.update(progressText)
       }
      }

      runOnUiThread {
       generationDialog.update(
        "Вариант ${i+1}/8 • seed $seed\nОцениваю сходство голоса…"
       )
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
      "VARIANT ${i+1}/8 • seed $seed\n"+
       pair.first+
       "\nVoice similarity: "+
       String.format(java.util.Locale.US,"%.4f",pair.second)
     )
    }

    runOnUiThread {
     setGenerationBusy(false)

     if(failure==null) {
      val rankedIndices=similarities.indices.sortedByDescending { similarities[it] }
      val bestIndex=rankedIndices.first()
      val top3=rankedIndices.take(3)

      lastReport=reports.joinToString("\n\n====================\n\n")
      refreshResults(names,names[bestIndex])

      copyReport.isEnabled=true
      play.isEnabled=true
      saveWav.isEnabled=true
      shareWav.isEnabled=true

      status.text=buildString {
       append("STAGE 3H SUCCESS — 8 variants ranked by voice similarity\n\n")
       for((rank,index) in rankedIndices.withIndex()) {
        append("${rank+1}. seed ${seeds[index]} • similarity ")
        append(String.format(java.util.Locale.US,"%.4f",similarities[index]))
        if(rank<3) append("  ★ TOP ${rank+1}")
        append("\n")
       }
       append("\nBest speaker similarity: seed ${seeds[bestIndex]}")
       append("\nTop 3 are shown in the generation window.")
       append("\nAll 8 WAV files remain available in the main result list.")
      }

      generationDialog.complete(
       top3.map { index ->
        GenerationProgressDialog.ResultItem(
         names[index],
         seeds[index],
         similarities[index]
        )
       }
      )
     } else {
      lastReport=reports.joinToString("\n\n")
      val error="Stage 3H voice search failed: ${failure?.javaClass?.simpleName}: ${failure?.message}"
      status.text=error
      generationDialog.fail(error)
     }
    }
   }.start()
  }

  profileSearch.setOnClickListener {
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

  multiReference.setOnClickListener {
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

  smoothHybrid.setOnClickListener {
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

  longSynthesis.setOnClickListener {
   val typed=inputText.text?.toString().orEmpty()
   if(typed.isBlank()) {
    status.text="Stage 3L: enter long Russian text first."
    return@setOnClickListener
   }
   val seed=seedInput.text?.toString()?.trim()?.toIntOrNull()
   if(seed==null) {
    status.text="Stage 3L: seed must be a whole number."
    return@setOnClickListener
   }

   val preset=qualityMode.selectedItemPosition
   val generationDialog=GenerationProgressDialog(this,filesDir)
   generationDialog.show()
   setGenerationBusy(true)
   lastReport=""
   status.text="Stage 3L: preparing long synthesis…"

   Thread {
    val result=runCatching {
     XttsLongSynthesis(filesDir).run(
      text=typed,
      preset=preset,
      seed=seed
     ) { message ->
      runOnUiThread {
       status.text=message
       generationDialog.update(message)
      }
     }
    }

    runOnUiThread {
     setGenerationBusy(false)
     result.onSuccess { longResult ->
      lastReport=longResult.report
      refreshResults(
       listOf(longResult.outputFile.name),
       longResult.outputFile.name
      )
      copyReport.isEnabled=true
      play.isEnabled=true
      saveWav.isEnabled=true
      shareWav.isEnabled=true
      status.text=longResult.report

      val score=if(longResult.similarity.isFinite()) longResult.similarity else 0.0
      generationDialog.complete(
       listOf(
        GenerationProgressDialog.ResultItem(
         fileName=longResult.outputFile.name,
         seed=seed,
         similarity=score,
         displayName="Длинная озвучка • ${longResult.chunks.size} фрагм."
        )
       )
      )
     }.onFailure {
      lastReport=""
      val error="Stage 3L failed: ${it.javaClass.simpleName}: ${it.message}"
      status.text=error
      generationDialog.fail(error)
     }
    }
   }.start()
  }

  referenceLab.setOnClickListener {
   val longRef=File(filesDir,"reference_long.wav")
   if(!longRef.exists()) {
    status.text="Reference Lab: сначала импортируй длинную чистую WAV-запись Shorts."
    return@setOnClickListener
   }

   val generationDialog=GenerationProgressDialog(this,filesDir)
   generationDialog.show()
   setGenerationBusy(true)
   lastReport=""
   status.text="Reference Lab: анализирую реальный голос…"

   Thread {
    val result=runCatching {
     XttsReferenceLab(filesDir).run(longRef) { message ->
      runOnUiThread {
       status.text=message
       generationDialog.update(message)
      }
     }
    }

    runOnUiThread {
     setGenerationBusy(false)
     result.onSuccess { lab ->
      lastReport=lab.report
      val names=lab.top5.map { it.file.name }
      refreshResults(names,lab.top5.first().file.name)
      copyReport.isEnabled=true
      play.isEnabled=true
      saveWav.isEnabled=true
      shareWav.isEnabled=true
      status.text=lab.report

      generationDialog.complete(
       lab.top5.map { c ->
        GenerationProgressDialog.ResultItem(
         fileName=c.file.name,
         seed=0,
         similarity=c.centroidSimilarity,
         displayName=
          "Ref ${c.index} • ${String.format(java.util.Locale.US,"%.1f",c.startSec)} с"+
          " • речь ${String.format(java.util.Locale.US,"%.0f",c.speechRatio*100)}%"
        )
       }
      )
     }.onFailure {
      lastReport=""
      val error="Reference Lab failed: ${it.javaClass.simpleName}: ${it.message}"
      status.text=error
      generationDialog.fail(error)
     }
    }
   }.start()
  }

  play.setOnClickListener {
   val selected=resultMode.selectedItem?.toString().orEmpty()
   val wav=File(filesDir,selected)
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
  }

  shareWav.setOnClickListener {
   val selected=resultMode.selectedItem?.toString().orEmpty()
   val wav=File(filesDir,selected)
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
 }

 override fun onDestroy() {
  player?.release()
  player=null
  super.onDestroy()
 }

 override fun onActivityResult(requestCode:Int,resultCode:Int,data:Intent?) {
  super.onActivityResult(requestCode,resultCode,data)
  if(resultCode!=Activity.RESULT_OK) return
  val u=data?.data?:return

  if(requestCode==10) {
   val f=File(filesDir,"reference_long.wav")
   runCatching {
    contentResolver.openInputStream(u)!!.use { input ->
     f.outputStream().use { output -> input.copyTo(output) }
    }
   }.onSuccess {
    lastReport=""
    status.text="Long Shorts WAV imported: ${f.length()/1024} KB\nReady for Stage 3I profile search."
   }.onFailure {
    status.text="Long Shorts WAV import failed: ${it.message}"
   }
   return
  }

  if(requestCode==9) {
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
   val f=File(filesDir,"reference.wav")
   contentResolver.openInputStream(u)!!.use { a -> f.outputStream().use { b -> a.copyTo(b) } }
   refPath=f.absolutePath
   File(filesDir,"conditioning_cache").deleteRecursively()
   lastReport=""
   status.text="Reference imported: ${f.length()/1024} KB\nConditioning cache cleared. Run Stage 3A."
  }

  if(requestCode==8) {
   status.text="Importing model archive…"
   Thread {
    val result=runCatching { importModelZip(u); validateModels() }
    runOnUiThread {
     status.text=result.fold(
      { "ONNX check:\n$it" },
      { "ONNX check failed: ${it.message}" }
     )
    }
   }.start()
  }
 }

 private fun importModelZip(uri:android.net.Uri) {
  modelDir.mkdirs()
  val base=modelDir.canonicalFile
  contentResolver.openInputStream(uri)!!.use { input ->
   ZipInputStream(input).use { zip ->
    var entry=zip.nextEntry
    while(entry!=null) {
     if(!entry.isDirectory) {
      val name=entry.name.replace('\\','/')
      val dest=File(base,name).canonicalFile
      require(dest.path.startsWith(base.path+File.separator)) { "Unsafe archive path" }
      dest.parentFile?.mkdirs()
      dest.outputStream().use { zip.copyTo(it) }
     }
     zip.closeEntry()
     entry=zip.nextEntry
    }
   }
  }
 }

 private fun validateModels():String {
  val models=modelDir.walkTopDown().filter { it.isFile && it.extension.equals("onnx",true) }.toList()
  require(models.isNotEmpty()) { "No .onnx files in imported ZIP" }
  val required=setOf("conditioning_encoder.onnx","speaker_encoder.onnx","gpt_model_int8.onnx","hifigan_vocoder.onnx")
  val missing=required-models.map { it.name }.toSet()
  require(missing.isEmpty()) { "Missing models: ${missing.joinToString()}" }
  val env=OrtEnvironment.getEnvironment()
  val opts=OrtSession.SessionOptions()
  try {
   return models.sortedBy { it.name }.joinToString("\n") { model ->
    env.createSession(model.absolutePath,opts).use { session ->
     "${model.name}: ${session.inputNames.size} inputs / ${session.outputNames.size} outputs"
    }
   }
  } finally { opts.close() }
 }
}
