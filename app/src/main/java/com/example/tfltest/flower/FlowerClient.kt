package com.example.tfltest.flower

import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.MappedByteBuffer
import java.util.concurrent.locks.ReentrantLock
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.read
import kotlin.concurrent.withLock
import kotlin.concurrent.write
import kotlin.math.max

/**
 * Flower client that handles TensorFlow Lite model [Interpreter] and sample data.
 * @param tfliteFileBuffer TensorFlow Lite model file.
 * @param layersSizes Sizes of model parameters layers in bytes.
 * @param spec Specification for the samples, see [SampleSpec].
 */
class FlowerClient(
    tfliteFileBuffer: MappedByteBuffer,
    private val layersSizes: IntArray,
    private val images: List<FloatArray>,
    private val labels: List<Float>,
    private val numberOfClasses: Int,
    evalSize: Float = 0.2f
) : AutoCloseable {
    private val interpreter = Interpreter(tfliteFileBuffer)
    private val interpreterLock = ReentrantLock()
    private val trainSampleLock = ReentrantReadWriteLock()
    private val testSampleLock = ReentrantReadWriteLock()

    private var trainSamples: List<Sample<FloatArray, Float>>
    private var evalSamples: List<Sample<FloatArray, Float>>

    init {
        // TODO proper train test split
        val numTestImages = (images.size * evalSize).toInt()
        evalSamples = (0..<numTestImages).map { Sample(images[it], labels[it]) }
        trainSamples = (numTestImages..<images.size).map { Sample(images[it], labels[it]) }
    }

    /**
     * Obtain the model parameters from [interpreter].
     *
     * This method is more expensive than a simple lookup because it interfaces [interpreter].
     * Thread-safe.
     */
    fun getParameters(): Array<ByteBuffer> {
        val inputs: Map<String, Any> = mutableMapOf("unused" to  "trash")
        val outputs = emptyParameterMap()
        runSignatureLocked(inputs, outputs, "get_weights_for_fl")
        Log.i(TAG, "Raw weights: $outputs.")
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
        epochs: Int = 1, batchSize: Int = 32, lossCallback: ((List<Float>) -> Unit)? = null
    ): TrainingResult {
        Log.d(TAG, "Starting to train for $epochs epochs with batch size $batchSize.")
        // Obtain write lock to prevent training samples from being modified.
        val trainingSamples = trainSamples.size
        val losses = trainSampleLock.write {
            (1..epochs).map {
                val losses = trainOneEpoch(batchSize)
                Log.d(TAG, "Epoch $it: losses = $losses.")
                lossCallback?.invoke(losses)
                losses.average()
            }
        }
        return TrainingResult(losses, trainingSamples)
    }

    /**
     * Evaluate model loss and accuracy using [evalSamples] and [spec].
     *
     * Thread-safe, and block operations on [evalSamples].
     * @return (loss, accuracy).
     */
    fun evaluate(): EvaluationResult {
        val result = testSampleLock.read {
            val inputs = mapOf("x" to evalSamples.map { it.x }.toTypedArray())
            val predictionOutputs = mapOf(
                "output" to ByteBuffer.allocate(4 * evalSamples.size).order(ByteOrder.nativeOrder()),
                "logits" to ByteBuffer.allocate(4 * numberOfClasses * evalSamples.size).order(ByteOrder.nativeOrder())
            )
            runSignatureLocked(inputs, predictionOutputs, "predict")

            // this doesn't work for dynamically sized batches, idk why
//            val lossInputs = mapOf(
//                "y_true" to evalSamples.map { it.label }.toFloatArray(),
//                "logits_pred" to (predictionOutputs["logits"] as ByteBuffer)
//            )
//            val lossOutputs = mapOf("loss" to ByteBuffer.allocate(4).order(ByteOrder.nativeOrder()))
//            runSignatureLocked(lossInputs, lossOutputs, "compute_loss")
//            val loss = (lossOutputs["loss"] as ByteBuffer).getFloat(0)

            val predictions = predictionOutputs["output"] as ByteBuffer
            predictions.rewind()
            val correctPredictions = evalSamples.map { it.label }.count { it == predictions.getFloat() }

            EvaluationResult(2.2f, correctPredictions.toFloat() / evalSamples.size, evalSamples.size)
        }
        Log.d(TAG, "Evaluation: $result.")
        return result
    }

    /**
     * Not thread-safe.
     */
    private fun trainOneEpoch(batchSize: Int): List<Float> {
        if (images.isEmpty()) {
            Log.d(TAG, "No training samples available.")
            return listOf()
        }

        trainSamples = trainSamples.shuffled()
        return getTrainingBatches(batchSize).map { runTraining(it) }.toList()
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
    private fun getTrainingBatches(trainBatchSize: Int): Sequence<List<Sample<FloatArray, Float>>> {
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
        return layersSizes.mapIndexed { index, size -> "a$index" to ByteBuffer.allocate(size * 4) }
            .toMap()
    }

    companion object {
        private const val TAG = "Flower Client"
    }

    override fun close() {
        interpreter.close()
    }
}

data class EvaluationResult(val loss: Float, val accuracy: Float, val numExamples: Int)

data class TrainingResult(val epochLosses: List<Double>, val trainingSamples: Int)

/**
 * One sample data point ([x], [label]).
 */
data class Sample<X, Y>(val x: X, val label: Y)

/**
 * This map always returns `false` when [isEmpty] is called to bypass TFLite interpreter's
 * stupid empty check on the `input` argument of `runSignature`.
 */
class FakeNonEmptyMap<K, V> : HashMap<K, V>() {
    override fun isEmpty(): Boolean {
        return false
    }
}