package com.example.ondevicetraining

import android.graphics.Color
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
import com.example.ondevicetraining.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    private lateinit var binding: ActivityMainBinding
    private var tfLiteModel = TFLiteModel(this)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)


        binding.drawView.run {
            setStrokeWidth(70.0f)
            setColor(Color.WHITE)
            setBackgroundColor(Color.BLACK)
        }

        binding.btnClear.setOnClickListener {
            binding.drawView.clearCanvas()
            binding.txtLabel.text = "Draw number!"
        }

        binding.drawView.setOnTouchListener { _, motionEvent ->

            binding.drawView.onTouchEvent(motionEvent)

            if (motionEvent.action == MotionEvent.ACTION_UP) {
                inference()
            }

            true
        }




        binding.btnInference.setOnClickListener {
            inference()
        }



        tfLiteModel.initialize().addOnFailureListener {
                e -> Log.e(TAG, "Error to setting up tflite $e")
        }

    }

    override fun onDestroy() {
        tfLiteModel.close()
        super.onDestroy()
    }


    fun inference() {
        val bitmap = binding.drawView.getBitmap()

        if ( tfLiteModel.isInit ){
            tfLiteModel
                .classifyAsync(bitmap)
                .addOnSuccessListener { result -> binding.txtLabel.text = result }
                .addOnFailureListener { e -> binding.txtLabel.text = "error $e" }
        }

    }
}