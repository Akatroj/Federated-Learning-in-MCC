package com.agh.federatedlearninginmcc.ocr

import android.util.Log
import com.agh.federatedlearninginmcc.ml.InferenceEngine
import com.agh.federatedlearninginmcc.dataset.OCRDataset
import com.agh.federatedlearninginmcc.ml.ModelVariant
import java.io.File
import java.time.Instant
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.time.DurationUnit
import kotlin.time.toDuration

class OCRService(
    private val localOCREngine: LocalOCREngine,
    private val cloudOCREngine: CloudOCREngine,
    private val inferenceEngine: InferenceEngine,
    private val ocrDataset: OCRDataset,
){
    private val localTimeInferenceInfo = mutableListOf<Pair<Float, Float>>()
    private val cloudComputationTimeInferenceInfo = mutableListOf<Pair<Float, Float>>()
    private val cloudTransmissionTimeInferenceInfo = mutableListOf<Pair<Float, Float>>()

    fun doOCR(img: File): String {
        val start = System.currentTimeMillis()

        val inferenceStart = System.currentTimeMillis()
        val imgInfo = ImageUtils.createImageInfo(img)
        val prediction = inferenceEngine.predictComputationTimes(imgInfo)
        Log.d(TAG, "Inference time: ${System.currentTimeMillis() - inferenceStart}ms, " +
                "Local cost: ${prediction.localTime.cost}, Cloud cost: ${prediction.cloudTime.cost}")

//        return if (prediction.shouldRunLocally) {
        return if (false) {
            Log.d(TAG, "running OCR locally")

            val ocrResult = localOCREngine.doOCR(img)
            val ocrTime = System.currentTimeMillis() - start

            Log.d(TAG, "OCR local time: $ocrTime, predicted: ${prediction.localTime.timeMs}")
            ocrDataset.addLocallyComputedTimeSample(imgInfo, ocrTime.toDuration(DurationUnit.MILLISECONDS))
            localTimeInferenceInfo.add(Pair(prediction.localTime.timeMs, ocrTime.toFloat()))
            ocrResult
        } else {
            val startInstant = Instant.now()
            Log.d(TAG, "running OCR remotely")

            val ocrResult = cloudOCREngine.doOCR(img)
            val totalTime = System.currentTimeMillis() - start
            val transmissionTime = totalTime.toDuration(DurationUnit.MILLISECONDS) - ocrResult.computationTime

            Log.d(TAG, "OCR remote total time: $totalTime, " +
                    "transmission time: $transmissionTime, " +
                    "predicted transmission time ${prediction.cloudTime.transmissionTimeMs}, " +
                    "computation time: ${ocrResult.computationTime}, " +
                    "predicted computation time: ${prediction.cloudTime.computationTimeMs}" )

            ocrDataset.addCloudComputedTimeSample(imgInfo, ocrResult.computationTime, transmissionTime, startInstant)
            cloudComputationTimeInferenceInfo.add(Pair(
                prediction.cloudTime.computationTimeMs, ocrResult.computationTime.inWholeMilliseconds.toFloat()))
            cloudTransmissionTimeInferenceInfo.add(Pair(
                prediction.cloudTime.transmissionTimeMs, transmissionTime.inWholeMilliseconds.toFloat()))

            ocrResult.text
        }
    }

    fun printStats() {
        val variants = mapOf(
            ModelVariant.LOCAL_TIME to localTimeInferenceInfo,
            ModelVariant.CLOUD_COMPUTATION_TIME to cloudComputationTimeInferenceInfo,
            ModelVariant.CLOUD_TRANSMISSION_TIME to cloudTransmissionTimeInferenceInfo
        )

        variants.entries.filter { it.value.isNotEmpty() }.forEach { entry ->
            var squaredError = 0.0f
            var absError = 0.0f
            entry.value.forEach { (yPred, yTrue) ->
                squaredError += (yPred - yTrue).pow(2)
                absError += abs(yPred - yTrue)
            }
            val rmse = sqrt(squaredError / entry.value.size)
            val mae = absError / entry.value.size
            Log.i(TAG, "Model: ${entry.key.name} Stats: samples=${entry.value.size} rmse=${rmse} mae=${mae}")
        }
    }

    companion object {
        private const val TAG = "OCRService"
    }
}
