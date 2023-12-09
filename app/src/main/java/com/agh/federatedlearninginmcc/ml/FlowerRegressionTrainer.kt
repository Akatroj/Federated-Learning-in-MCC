package com.agh.federatedlearninginmcc.ml

import com.example.tfltest.flower.Sample
import com.example.tfltest.flower.TrainingResult
import java.nio.ByteBuffer

class FlowerRegressionTrainer(
    private val model: FlowerRegressionModel,
    private val xs: List<FloatArray>,
    private val ys: FloatArray,
    private val evalSize: Float = 0.2f
) {
    private var trainSamples: List<Sample<FloatArray, Float>>
    private var evalSamples: List<Sample<FloatArray, Float>>

    init {
        // TODO proper train eval split
        val numTestImages = (xs.size * evalSize).toInt()
        evalSamples = (0..<numTestImages).map { Sample(xs[it], ys[it]) }
        trainSamples = (numTestImages..<xs.size).map { Sample(xs[it], ys[it]) }
    }

    fun updateParameters(parameters: Array<ByteBuffer>): Array<ByteBuffer> = model.updateParameters(parameters)

    fun fit(epochs: Int, batchSize: Int): TrainingResult = model.fit(trainSamples, epochs, batchSize)

    fun evaluate(): RegressionEvaluationResult = model.evaluate()

    fun getParameters(): Array<ByteBuffer> = model.getParameters()
}
