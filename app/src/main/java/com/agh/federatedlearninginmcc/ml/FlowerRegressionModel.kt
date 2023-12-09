package com.agh.federatedlearninginmcc.ml

import android.util.Log
import com.example.tfltest.flower.Sample
import com.example.tfltest.flower.TrainingResult
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.MappedByteBuffer
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.math.max

class FlowerRegressionModel(
    tfliteFileBuffer: MappedByteBuffer,
    val inputDimensions: Int,
    private val modelTag: String,
    private val layerSizes: IntArray,
    private val xMeans: FloatArray, // TODO handle these means/stds differently
    private val xStds: FloatArray,
    private val yMean: Float,
    private val yStd: Float
) : AutoCloseable {
    private val interpreter = Interpreter(tfliteFileBuffer)
    private val interpreterLock = ReentrantLock()

    fun predict(x: FloatArray): Float {
        val normalizedX = x.copyOf()
        normalizedX.forEachIndexed { i, value ->
            normalizedX[i] = (value - xMeans[i]) / xStds[i]
        }
        val inputs = mutableMapOf<String, Any>(
            "x" to Array(1) { x }
        )
        val outputs = mutableMapOf<String, Any>(
            "output" to ByteBuffer.allocate(4).order(ByteOrder.nativeOrder())
        )
        runSignatureLocked(inputs, outputs, "predict")
        val res = outputs["output"] as ByteBuffer
        val y = res.getFloat(0)
        return y * yStd + yMean
    }

    /**
     * Obtain the model parameters from [interpreter].
     *
     * This method is more expensive than a simple lookup because it interfaces [interpreter].
     * Thread-safe.
     */
    fun getParameters(): Array<ByteBuffer> {
        val inputs: Map<String, Any> = mutableMapOf("unused" to "trash")
        val outputs = emptyParameterMap()
        runSignatureLocked(inputs, outputs, "get_weights_for_fl")
        Log.i(modelTag, "Raw weights: $outputs.")
        return parametersFromMap(outputs)
    }

    /**
     * Update the model parameters in [interpreter] with [parameters].
     *
     * This method is more expensive than a simple "set" because it interfaces [interpreter].
     * Thread-safe.
     */
    fun updateParameters(parameters: Array<ByteBuffer>): Array<ByteBuffer> {
        val outputs = emptyParameterMap()
        runSignatureLocked(parametersToMap(parameters), outputs, "set_weights_from_fl")
        return parametersFromMap(outputs)
    }

    /**
     * Fit the local model using [trainingSamples] for [epochs] epochs with batch size [batchSize].
     *
     * Thread-safe, and block operations on [trainingSamples].
     * @param lossCallback Called after every epoch with the [List] of training losses.
     * @return [List] of average training losses for each epoch.
     */
    fun fit(
        trainSamples: List<Sample<FloatArray, Float>>,
        epochs: Int = 1,
        batchSize: Int = 3
    ): TrainingResult {
        Log.d(modelTag, "Starting to train for $epochs epochs with batch size $batchSize.")

        val trainingSamples = trainSamples.size
        val losses = (1..epochs).map { epoch ->
            val losses = getTrainingBatches(trainSamples.shuffled(), batchSize).map { runTraining(it) }.toList()
            Log.d(modelTag, "Epoch $epoch: losses = $losses.")
            losses.average()
        }

        return TrainingResult(losses, trainingSamples)
    }

    /**
     * Evaluate model loss and accuracy using [evalSamples] and [spec].
     *
     * Thread-safe, and block operations on [evalSamples].
     * @return (loss).
     */
    fun evaluate(): RegressionEvaluationResult {
        // TODO
        return RegressionEvaluationResult(2.2f, 0)
    }

    /**
     * Not thread-safe because we assume [trainSampleLock] is already acquired.
     */
    private fun runTraining(samples: List<Sample<FloatArray, Float>>): Float {
        val inputs = mapOf<String, Any>(
            "x_batch" to samples.map { it.x }.toTypedArray(),
            "y_batch" to samples.map { it.label }.toFloatArray(),
        )
        val loss = FloatBuffer.allocate(1)
        val outputs = mapOf<String, Any>(
            "loss" to loss,
        )
        runSignatureLocked(inputs, outputs, "train_epoch")
        return loss.get(0)
    }

    /**
     * Constructs an iterator that iterates over training sample batches.
     */
    private fun getTrainingBatches(
        trainSamples: List<Sample<FloatArray, Float>>,
        trainBatchSize: Int
    ): Sequence<List<Sample<FloatArray, Float>>> {
        return sequence {
            var nextIndex = 0

            while (nextIndex < trainSamples.size) {
                var fromIndex = nextIndex
                nextIndex += trainBatchSize

                if (nextIndex >= trainSamples.size) {
                    fromIndex = max(0, trainSamples.size - trainBatchSize)
                    nextIndex = trainSamples.size
                }

                yield(trainSamples.subList(fromIndex, nextIndex))
            }
        }
    }

    private fun parametersFromMap(map: Map<String, Any>): Array<ByteBuffer> {
        return (0 until map.size).map {
            val buffer = map["a$it"] as ByteBuffer
            buffer.rewind()
            buffer
        }.toTypedArray()
    }

    private fun parametersToMap(parameters: Array<ByteBuffer>): Map<String, Any> {
        return parameters.mapIndexed { index, bytes -> "a$index" to bytes }.toMap()
    }

    private fun runSignatureLocked(
        inputs: Map<String, Any>,
        outputs: Map<String, Any>,
        signatureKey: String
    ) {
        interpreterLock.withLock {
            interpreter.runSignature(inputs, outputs, signatureKey)
        }
    }

    private fun emptyParameterMap(): Map<String, Any> {
        return layerSizes.mapIndexed { index, size ->
                "a$index" to ByteBuffer.allocate(size * 4)
        }.toMap()
    }

    override fun close() {
        interpreter.close()
    }
}

data class RegressionEvaluationResult(val loss: Float, val numExamples: Int)
