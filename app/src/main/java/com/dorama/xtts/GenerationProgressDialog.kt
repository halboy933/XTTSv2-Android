package com.dorama.xtts

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
  progress.isIndeterminate=true
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
  if(dialog.isShowing) message.text=text
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
