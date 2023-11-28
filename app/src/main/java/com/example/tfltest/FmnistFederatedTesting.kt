package com.example.tfltest

import android.content.Context
import android.graphics.BitmapFactory
import android.util.Log
import com.agh.federatedlearninginmcc.flower.FlowerClient
import com.agh.federatedlearninginmcc.flower.createFlowerService
import kotlinx.coroutines.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.IntBuffer

class FmnistFederatedTesting(
    private val context: Context,
    private val serverIp: String,
    private val port: Int
) {
    private lateinit var client: FlowerClient
    private val layerSizes = intArrayOf(784 * 128, 128, 128 * 10, 10)

     suspend fun connect() = coroutineScope {
         Log.d(TAG, "Creating flower client")
         val modelFile = FileUtil.loadMappedFile(context, MODEL_ASSETS_FILE)
         client = FlowerClient(modelFile, layerSizes, loadImages(), loadLabels())

         Log.d(TAG, "Running server")
         createFlowerService(serverIp, port, false, client) { Log.d("${TAG}_grpc", it) }
         Log.d(TAG, "Exited")
    }

    fun testTensorflowPrediction(interpreter: Interpreter, imgIdToPredict: Int, printLogits: Boolean = false) {
        val trueLabel = loadLabels()[imgIdToPredict]
        val imagePixels = loadImages()[imgIdToPredict]

        val inputs: MutableMap<String, Any> = HashMap()
        inputs["x"] = imagePixels

        val outputs: MutableMap<String, Any> = HashMap()
        outputs["output"] = ByteBuffer.allocate(4).order(ByteOrder.nativeOrder())
        outputs["logits"] = TensorBuffer.createFixedSize(intArrayOf(1, 10), DataType.FLOAT32).buffer

        interpreter.runSignature(inputs, outputs, "predict")
        Log.d(TAG, "finished prediction")

        if (printLogits) {
            val logits = outputs["logits"] as ByteBuffer
            logits.rewind()
            val logitsRes: MutableList<Float> = mutableListOf()
            for (i in 0 until 10) {
                logitsRes.add(logits.getFloat())
            }
            Log.d(TAG, logitsRes.toString())
        }

        val classificationBuff = outputs["output"] as ByteBuffer
        val predictedLabel = classificationBuff.getInt(0)

        Log.d(TAG, "true: $trueLabel predicted: $predictedLabel")
    }

     fun testTensorflowLearning(interpreter: Interpreter, datasetSize: Int = 100, epochs: Int = 1) {
        val labels = loadLabels()
        val images = loadImages(datasetSize)

        Log.d(TAG, "Loaded data")

        for (epoch in 0 until epochs) {
            val labelsBatch = FloatArray(TRAIN_BATCH_SIZE)
            val imgBatch: ArrayList<FloatArray> = ArrayList(TRAIN_BATCH_SIZE)

            var loss = 0.0f
            for (i in 0 until TRAIN_BATCH_SIZE) {
                imgBatch.add(images[i])
            }

            for (batch in 0 until  datasetSize / TRAIN_BATCH_SIZE) {
                for (i in 0  until TRAIN_BATCH_SIZE) {
                    labelsBatch[i] = labels[i + batch * TRAIN_BATCH_SIZE]
                    imgBatch[i] = images[i + batch * TRAIN_BATCH_SIZE]
                }

                val inputs: MutableMap<String, Any> = HashMap()
                val outputs: MutableMap<String, Any> = HashMap()

                inputs["x_batch"] = imgBatch.toTypedArray()
                inputs["y_batch"] = labelsBatch
                outputs["loss"] = FloatBuffer.allocate(1)

                interpreter.runSignature(inputs, outputs, "train_epoch")

                val lossBuff = outputs["loss"] as FloatBuffer

                Log.d(TAG, "Trained batch $batch")
                val batchLoss = lossBuff.get(0)
                Log.d(TAG, "Loss: $batchLoss")
                loss += batchLoss
            }
            Log.d(TAG, "Average loss: ${loss / (datasetSize / TRAIN_BATCH_SIZE)}")
        }
    }

    fun loadModel(tryRestoring: Boolean): Interpreter {
        val options = Interpreter.Options()
        val modelFile = FileUtil.loadMappedFile(context, MODEL_ASSETS_FILE)
        options.numThreads = 1
        Log.d(TAG, "loading interpreter")
        val interpreter = Interpreter(modelFile, options)
        if (tryRestoring) {
            val inputs: MutableMap<String, Any> = mutableMapOf()
            val modelEditedFile = File(context.filesDir, MODEL_EDIT_FILE)
            if (modelEditedFile.exists()) {
                inputs["path"] = modelEditedFile.absolutePath
                val outputs: MutableMap<String, Any> = mutableMapOf()
                outputs["result"] = IntBuffer.allocate(1)
                interpreter.runSignature(inputs, outputs, "restore")
                Log.d(TAG, "Restored model")
            } else {
                Log.d(TAG, "Model file doesn't exist, not restoring")
            }
        }
        return interpreter
    }

    fun writeModel(modelInterpreter: Interpreter)  {
        val inputs: MutableMap<String, Any> = mutableMapOf()
        inputs["path"] = File(context.filesDir, MODEL_EDIT_FILE).absolutePath
        val outputs: MutableMap<String, Any> = mutableMapOf()
        outputs["result"] = IntBuffer.allocate(1)
        modelInterpreter.runSignature(inputs, outputs, "save")
        val buff = outputs["result"] as IntBuffer
        buff.rewind()
        Log.d(TAG, "Written, res=${buff.get()}")
    }

    private fun loadImages(numImgs: Int = 100): List<FloatArray> {
        val imgPixelsBuff = IntArray(IMG_SIZE)
        return (0..<numImgs)
            .map { getInputFilePath(it) }
            .map { path -> context.assets.open(path).use { BitmapFactory.decodeStream(it) } }
            .map { bitmap ->
                val imgGrayscale = FloatArray(IMG_SIZE)
                bitmap.getPixels(imgPixelsBuff, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
                for (j in 0 ..< IMG_SIZE) {
                    imgGrayscale[j] = convertToGrayscaleFloat(imgPixelsBuff[j])
                }
                imgGrayscale
            }.toList()
    }

    private fun loadLabels(): List<Float> {
        // for simplicity, images for these labels are stored in assets
        return arrayListOf(
            9, 2, 1, 1, 6, 1, 4, 6, 5, 7, 4, 5, 7, 3, 4, 1, 2, 4, 8, 0, 2, 5, 7, 9, 1,
            4, 6, 0, 9, 3, 8, 8, 3, 3, 8, 0, 7, 5, 7, 9, 6, 1, 3, 7, 6, 7, 2, 1, 2, 2,
            4, 4, 5, 8, 2, 2, 8, 4, 8, 0, 7, 7, 8, 5, 1, 1, 2, 3, 9, 8, 7, 0, 2, 6, 2,
            3, 1, 2, 8, 4, 1, 8, 5, 9, 5, 0, 3, 2, 0, 6, 5, 3, 6, 7, 1, 8, 0, 1, 4, 2
        ).map { it.toFloat() }
    }

    private fun convertToGrayscaleFloat(color: Int): Float {
        return ((color shr 16 and 0xFF) * 0.299f + (color shr 8 and 0xFF) * 0.587f + (color and 0xFF) * 0.114f) / 255.0f
    }

    private fun getInputFilePath(fileNum: Int, format: String = "png"): String = "$DATASET_PATH/x_$fileNum.$format"

    companion object Constants {
        const val TAG = "FmnistTesting"
        const val IMG_WIDTH = 28
        const val IMG_HEIGHT = 28
        const val IMG_SIZE = IMG_WIDTH * IMG_HEIGHT
        const val TRAIN_BATCH_SIZE = 10
        const val DATASET_PATH = "fmnist_images"
        // tflite model file generated by python script should be placed in assets with this name
        const val MODEL_ASSETS_FILE = "model.tflite"
        // trained model should be saved to/loaded from this file
        const val MODEL_EDIT_FILE = "model.edit.tflite"
    }
}
