package com.agh.federatedlearninginmcc.ml

import android.util.Log
import com.example.tfltest.flower.Sample
import com.example.tfltest.flower.TrainingResult
import java.nio.ByteBuffer

class FlowerRegressionTrainer(
    private val model: FlowerRegressionModel,
    private val modelEditPath: String,
    private val xs: List<FloatArray>,
    private val ys: FloatArray,
    private val evalSize: Float = 0.2f,
    private val evalBatchSize: Int = 4
) {
    private var trainSamples: List<Sample<FloatArray, Float>>
    private var evalSamples: List<Sample<FloatArray, Float>>
    private var firstRound = true

    init {
        // TODO proper train eval split
        val numEvalImages = (xs.size * evalSize).toInt()
        evalSamples = (0..<numEvalImages).map { Sample(xs[it], ys[it]) }
        trainSamples = (numEvalImages..<xs.size).map { Sample(xs[it], ys[it]) }
    }

    fun updateParameters(parameters: Array<ByteBuffer>): Array<ByteBuffer> {
        val params = model.updateParameters(parameters)
        try {
            model.saveToDisk(modelEditPath)
            Log.d(TAG, "Saved model to disk")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save updated model to disk", e)
        }
        return params
    }

    fun fit(epochs: Int, batchSize: Int): TrainingResult {
        if (firstRound) {
            // to have loss before first training round
            firstRound = false
            return TrainingResult(epochLosses = listOf(0.0), trainingSamples=trainSamples.size)
        }
        return model.fit(trainSamples, epochs, batchSize)
    }

    fun evaluate(): RegressionEvaluationResult = model.evaluate(evalSamples, evalBatchSize)

    fun getParameters(): Array<ByteBuffer> = model.getParameters()

    companion object {
        private const val TAG = "FlowerRegressionTrainer"
    }
}
