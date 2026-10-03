package com.dorama.xtts

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

class XttsVoiceSimilarity(private val filesDir:File) {
 private val referenceEmbeddingFile=File(filesDir,"conditioning_cache/speaker_embedding.f32")

 fun scoreAgainstEmbedding(reference:FloatArray,generatedWav:File):Double {
  require(reference.size==512) { "Reference speaker embedding invalid: ${reference.size}" }
  val generated=XttsConditioning(filesDir).computeSpeakerEmbedding(generatedWav)
  require(generated.size==512) { "Generated speaker embedding invalid: ${generated.size}" }
  return cosineSimilarity(reference,generated)
 }

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
