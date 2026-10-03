from pathlib import Path

root = Path(".")
gradle = root / "app/build.gradle.kts"
lab = root / "app/src/main/java/com/dorama/xtts/XttsReferenceLab.kt"

for p in (gradle, lab):
    if not p.exists():
        raise SystemExit(f"Missing expected file: {p}")

# ---- bump version ----
s = gradle.read_text()
assert 'versionCode = 16' in s, "Expected Stage 3M versionCode 16"
assert 'versionName = "0.3.12-stage3m"' in s, "Expected Stage 3M versionName"
s = s.replace('versionCode = 16', 'versionCode = 17', 1)
s = s.replace('versionName = "0.3.12-stage3m"', 'versionName = "0.3.13-stage3m-hotfix"', 1)
gradle.write_text(s)

# ---- Reference Lab playback hotfix ----
s = lab.read_text()

start_marker = ''' fun run(longReference:File,progress:(String)->Unit):Result {
  require(longReference.exists()) { "reference_long.wav is missing." }

'''
assert start_marker in s, "ReferenceLab run() start marker not found"
start_replacement = ''' fun run(longReference:File,progress:(String)->Unit):Result {
  require(longReference.exists()) { "reference_long.wav is missing." }

  // Playback/Save/Share resolve candidate names directly under filesDir.
  // Remove stale root-level copies before a new Reference Lab run.
  filesDir.listFiles()
   ?.filter { it.isFile && it.name.matches(Regex("xtts_refcand_\\\\d{2}\\\\.wav")) }
   ?.forEach { runCatching { it.delete() } }

'''
s = s.replace(start_marker, start_replacement, 1)

rank_marker = '''  val top5=ranked.take(5)

  progress("Stage 3M 5/5: готово — текущий голос не изменён.")

'''
assert rank_marker in s, "ReferenceLab top5 marker not found"
rank_replacement = '''  val top5=ranked.take(5)

  // Analysis candidates live in reference_lab_candidates/.
  // UI playback/save/share look in filesDir, so keep playable Top-5 copies there.
  val playableTop5=top5.map { candidate ->
   val playable=File(filesDir,candidate.file.name)
   candidate.file.copyTo(playable,overwrite=true)
   candidate.copy(file=playable)
  }

  progress("Stage 3M 5/5: готово — Top 5 сохранены для прослушивания.")

'''
s = s.replace(rank_marker, rank_replacement, 1)

report_marker = '''   for((rank,c) in top5.withIndex()) {
'''
assert report_marker in s, "ReferenceLab report top5 marker not found"
s = s.replace(report_marker, '''   for((rank,c) in playableTop5.withIndex()) {
''', 1)

return_marker = '''  return Result(candidates,top5,report)
'''
assert return_marker in s, "ReferenceLab return marker not found"
s = s.replace(return_marker, '''  return Result(candidates,playableTop5,report)
''', 1)

lab.write_text(s)

print("Stage 3M playback hotfix applied successfully")
print("Version: code 17 / 0.3.13-stage3m-hotfix")
print("Top-5 reference WAV files are copied to filesDir root")
print("Generation dialog playback can now find xtts_refcand_XX.wav")
print("Main Play / Save / Share use the same root-level files")
print("Old root-level xtts_refcand_XX.wav copies are cleared before each Reference Lab run")
print("Reference ranking / similarity logic is unchanged")
