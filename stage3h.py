from pathlib import Path

root = Path(".")
gradle = root / "app/build.gradle.kts"
main = root / "app/src/main/java/com/dorama/xtts/MainActivity.kt"
dialog = root / "app/src/main/java/com/dorama/xtts/GenerationProgressDialog.kt"

for p in (gradle, main, dialog):
    if not p.exists():
        raise SystemExit(f"Missing expected file: {p}")

s = gradle.read_text()
assert 'versionCode = 10' in s, "Expected Stage 3G versionCode 10"
assert 'versionName = "0.3.6-stage3g"' in s, "Expected Stage 3G versionName"
s = s.replace('versionCode = 10', 'versionCode = 11')
s = s.replace('versionName = "0.3.6-stage3g"', 'versionName = "0.3.7-stage3h"')
gradle.write_text(s)

s = dialog.read_text()
s = s.replace('\\\\n', '\\n')
s = s.replace('text="Результаты"', 'text="Топ-3 по сходству"')
s = s.replace(
    'append("Готово. Выберите вариант и нажмите «Прослушать».\\n")',
    'append("Готово. Ниже лучшие варианты по speaker similarity.\\n")'
)
dialog.write_text(s)

s = main.read_text()
s = s.replace('\\\\n', '\\n')
s = s.replace('text="XTTS-v2 Android V2 • Stage 3G"', 'text="XTTS-v2 Android V2 • Stage 3H"')
s = s.replace(
    'text="Russian XTTS-v2 • voice similarity + generation window"',
    'text="Russian XTTS-v2 • 8-variant voice search + Top 3"'
)
s = s.replace(
    'val synthesize=Button(this).apply { text="Generate 1 variant (Stage 3G)" }',
    'val synthesize=Button(this).apply { text="Generate 1 variant (Stage 3H)" }'
)
s = s.replace(
    'val synthesize3=Button(this).apply { text="Generate 3 variants (seed, seed+1, seed+2)" }',
    'val synthesize3=Button(this).apply { text="Find best voice • 8 variants → Top 3" }'
)
s = s.replace(
    'append("Ready for Stage 3A / 3B / 3C / 3D / 3E / 3F / 3G.")',
    'append("Ready for Stage 3A / 3B / 3C / 3D / 3E / 3F / 3G / 3H.")'
)

single_start = s.index('  synthesize.setOnClickListener {')
multi_start = s.index('  synthesize3.setOnClickListener {', single_start)
single_block = s[single_start:multi_start].replace('Stage 3G', 'Stage 3H')
s = s[:single_start] + single_block + s[multi_start:]

multi_start = s.index('  synthesize3.setOnClickListener {')
play_start = s.index('  play.setOnClickListener {', multi_start)

multi_handler = r'''  synthesize3.setOnClickListener {
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

'''

s = s[:multi_start] + multi_handler + s[play_start:]
main.write_text(s)

print("Stage 3H applied successfully")
print("Version: code 11 / 0.3.7-stage3h")
print("8 deterministic variants: base seed .. base seed+7")
print("Automatic speaker similarity ranking")
print("Generation window shows Top 3 only")
print("All 8 WAV files remain in the main result list")
print(r'Fixed literal \n display from Stage 3G')
