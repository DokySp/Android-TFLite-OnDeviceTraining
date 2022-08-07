package com.example.ondevicetraining

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.TaskCompletionSource
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


class TFLiteModel(private val context: Context) {

    companion object {
        private const val TAG = "DigitClassifier"
        private const val FLOAT_TYPE_SIZE = 4
        private const val PIXEL_SIZE = 1
        private const val OUTPUT_CLASSES_COUNT = 10
    }

    var isInit = false

    /** Executor to run inference task in the background. */
    private val executorService: ExecutorService = Executors.newCachedThreadPool()

    private var inputImageWidth: Int = 0 // infered by tflite
    private var inputImageHeight: Int = 0 // infered by tflite
    private var inputSize: Int = 28 // infered by tflite

    fun initialize(): Task<Void?> {
        val task = TaskCompletionSource<Void?>()
        executorService.execute {
            try{
                initInterpreter()
            } catch (e: IOException) {
                task.setException(e)
            }
        }

        return task.task
    }





    // 인터프리터 추가
    private var interpreter: Interpreter? = null


    @Throws(IOException::class)
    private fun initInterpreter() {
        // 인터프리터 초기화
        val assetManager = context.assets
        val model = loadModelFile(assetManager, "mnist.tflite")
        val interpreter = Interpreter(model)

        val inputShape = interpreter.getInputTensor(0).shape()
        // (1, 28, 28)
        inputImageWidth = inputShape[1]
        inputImageHeight = inputShape[2]

        // pixel size == channel size
        inputSize = FLOAT_TYPE_SIZE * inputImageWidth * inputImageHeight * PIXEL_SIZE

        this.interpreter = interpreter

        // 마무리
        isInit = true
        Log.d(TAG, "initInterpreter() true")
    }



    @Throws(IOException::class)
    private fun loadModelFile(assetManager: AssetManager, filename: String): ByteBuffer {

        val fileDescriptor = assetManager.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }






    private fun classify(bitmap: Bitmap): String {
        check(isInit) { "TF Lite Interpreter is not initialized yet." }

        val resizedImg = Bitmap.createScaledBitmap(
            bitmap,
            inputImageWidth,
            inputImageHeight,
            true
        )
        val byteBuffer = convertBitmapToByteBuffer(resizedImg)

        val output = Array(1) { FloatArray(OUTPUT_CLASSES_COUNT) }

        interpreter?.run(byteBuffer, output)

        // (1,10)
        val resRaw = output[0]

        val resHash = HashMap<Int, Float>()

        var ii = 0
        resRaw.forEach {
            resHash[ii++] = it
        }

        // Top 10 result
        val result = resHash.toList().sortedBy { (_, value) -> value }.reversed().toMap()

        return "It might be ${result.keys.toList()[0]}, or ${result.keys.toList()[1]}"
    }



    fun classifyAsync(bitmap: Bitmap): Task<String> {
        val task = TaskCompletionSource<String>()
        executorService.execute {
            val result = classify(bitmap)
            task.setResult(result)
        }
        return task.task
    }






    fun close() {
        executorService.execute {

            interpreter?.close()

            Log.d(TAG, "Closed TFLite interpreter.")
        }
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(inputSize)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputImageWidth * inputImageHeight)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        for (pixelValue in pixels) {
            val r = (pixelValue shr 16 and 0xFF)
            val g = (pixelValue shr 8 and 0xFF)
            val b = (pixelValue and 0xFF)

            // Convert RGB to grayscale and normalize pixel value to [0..1].
            val normalizedPixelValue = (r + g + b) / 3.0f / 255.0f
            byteBuffer.putFloat(normalizedPixelValue)
        }

        return byteBuffer
    }
}



