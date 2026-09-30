from pathlib import Path

p = Path("app/src/main/java/com/dorama/xtts/MainActivity.kt")

s = p.read_text(encoding="utf-8")

s = s.replace(
    'text="XTTS-v2 Android V1"',
    'text="XTTS-v2 Android V2 • Stage 2"'
)

s = s.replace(
    'val generate = Button(this).apply { setText("Generate WAV"); isEnabled = false }',
    'val generate = Button(this).apply { setText("Inspect ONNX interfaces"); isEnabled = true }'
)

s = s.replace(
    'generate.setOnClickListener { status.text = "Full XTTS inference is not implemented in stage 1" }',
'''generate.setOnClickListener {
    status.text = "Opening ONNX models..."
    Thread {
      val result = runCatching { inspectModels() }
      runOnUiThread {
        status.text = result.fold(
          { it },
          { "ONNX inspect failed: ${it.message}" }
        )
      }
    }.start()
  }'''
)

marker = " private fun validateModels():String {"

inspect = r'''
 private fun inspectModels():String {
  val models=modelDir.walkTopDown()
    .filter { it.isFile && it.extension.equals("onnx",true) }
    .toList()

  require(models.isNotEmpty()) {
    "Models are missing. Import model ZIP first."
  }

  val env=OrtEnvironment.getEnvironment()
  val opts=OrtSession.SessionOptions()

  return try {
    buildString {
      models.sortedBy { it.name }.forEach { model ->

        env.createSession(model.absolutePath,opts).use { session ->

          append("\n[")
          append(model.name)
          append("]\n")

          append("IN:\n")

          session.inputInfo.entries.forEach { (name,info) ->
            append("  ")
            append(name)
            append(" : ")
            append(info.info.toString())
            append("\n")
          }

          append("OUT:\n")

          session.outputInfo.entries.forEach { (name,info) ->
            append("  ")
            append(name)
            append(" : ")
            append(info.info.toString())
            append("\n")
          }
        }
      }
    }
  } finally {
    opts.close()
  }
 }

'''

if "private fun inspectModels()" not in s:
    if marker not in s:
        raise SystemExit("ERROR: validateModels marker not found")

    s = s.replace(marker, inspect + marker)

p.write_text(s, encoding="utf-8")

print("Stage 2 applied successfully")
print(p)