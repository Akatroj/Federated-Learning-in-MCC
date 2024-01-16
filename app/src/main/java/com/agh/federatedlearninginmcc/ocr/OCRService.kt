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

// not thread-safe
class OCRService(
    private val localOCREngine: LocalOCREngine,
    private val cloudOCREngine: CloudOCREngine,
    private val inferenceEngine: InferenceEngine,
    private val ocrDataset: OCRDataset,
    private val transmissionTestInfo: TransmissionTestInfo,
    private val saveNewSamples: Boolean = true
){
    private val localTimeInferenceInfo = mutableListOf<Pair<Float, Float>>()
    private val cloudComputationTimeInferenceInfo = mutableListOf<Pair<Float, Float>>()
    private val cloudTransmissionTimeInferenceInfo = mutableListOf<Pair<Float, Float>>()
    private val inferenceTimes = mutableListOf<Long>()

    private var currentNumNodes = transmissionTestInfo.nodes

    fun doOCR(img: File, forceLocalExecution: Boolean = false): OCRResultData {
        val start = System.currentTimeMillis()

        val inferenceStart = System.currentTimeMillis()
        val imgInfo = ImageUtils.createImageInfo(img)
        val prediction = inferenceEngine.predictComputationTimes(imgInfo, currentNumNodes, transmissionTestInfo.rttMs)
        val inferenceTime = System.currentTimeMillis() - inferenceStart
        inferenceTimes.add(inferenceTime)
        Log.d(TAG, "Inference time: ${inferenceTime}ms, " +
                "Local cost: ${prediction.localTime.cost}, Cloud cost: ${prediction.cloudTime.cost}")

        return if (forceLocalExecution || prediction.shouldRunLocally) {
            Log.d(TAG, "running OCR locally")

            val ocrResult = localOCREngine.doOCR(img)
            val ocrTime = System.currentTimeMillis() - start

            Log.d(TAG, "OCR local time: $ocrTime, predicted: ${prediction.localTime.timeMs}")
            if (saveNewSamples) {
                ocrDataset.addLocallyComputedTimeSample(
                    imgInfo,
                    ocrTime.toDuration(DurationUnit.MILLISECONDS)
                )
            }
            localTimeInferenceInfo.add(Pair(prediction.localTime.timeMs, ocrTime.toFloat()))
            OCRResultData(ocrResult, prediction, ocrTime.toFloat())
        } else {
            val startInstant = Instant.now()
            Log.d(TAG, "running OCR remotely")

            val ocrResult: CloudOCRInfo
            try {
                ocrResult = cloudOCREngine.doOCR(img)
                currentNumNodes = ocrResult.numNodes
            } catch (e: Exception) {
                // this may happen e.g. when pod dies due to overload
                Log.e(TAG, "cloud OCR failed, retrying locally", e)
                return doOCR(img, forceLocalExecution=true)
            }

            val totalTime = System.currentTimeMillis() - start
            val transmissionTime = totalTime.toDuration(DurationUnit.MILLISECONDS) - ocrResult.computationTime

            Log.d(TAG, "OCR remote total time: $totalTime, " +
                    "transmission time: $transmissionTime, " +
                    "predicted transmission time ${prediction.cloudTime.transmissionTimeMs}, " +
                    "computation time: ${ocrResult.computationTime}, " +
                    "predicted computation time: ${prediction.cloudTime.computationTimeMs}" )

            if (saveNewSamples) {
                ocrDataset.addCloudComputedTimeSample(
                    imgInfo, ocrResult.computationTime, transmissionTime,
                    startInstant, currentNumNodes, transmissionTestInfo.rttMs
                )
            }
            cloudComputationTimeInferenceInfo.add(Pair(
                prediction.cloudTime.computationTimeMs, ocrResult.computationTime.inWholeMilliseconds.toFloat()))
            cloudTransmissionTimeInferenceInfo.add(Pair(
                prediction.cloudTime.transmissionTimeMs, transmissionTime.inWholeMilliseconds.toFloat()))

            OCRResultData(ocrResult.text, prediction, prediction.cloudTime.computationTimeMs + transmissionTime.inWholeMilliseconds)
        }
    }

    fun printStatsAndGetSummary(): String {
        val variants = mapOf(
            ModelVariant.LOCAL_TIME to localTimeInferenceInfo,
            ModelVariant.CLOUD_COMPUTATION_TIME to cloudComputationTimeInferenceInfo,
            ModelVariant.CLOUD_TRANSMISSION_TIME to cloudTransmissionTimeInferenceInfo
        )

        val meanInferenceTime = if (inferenceTimes.size > 0) {
            inferenceTimes.sum().toFloat() / inferenceTimes.size
        } else {
            0
        }
        Log.i(TAG, "Mean inference time: $meanInferenceTime")

        return variants.entries.filter { it.value.isNotEmpty() }.map { entry ->
            var squaredError = 0.0f
            var absError = 0.0f
            entry.value.forEach { (yPred, yTrue) ->
                squaredError += (yPred - yTrue).pow(2)
                absError += abs(yPred - yTrue)
            }
            val rmse = sqrt(squaredError / entry.value.size)
            val mae = absError / entry.value.size
            Log.i(TAG, "Model: ${entry.key.name} Stats: samples=${entry.value.size} rmse=${rmse} mae=${mae}")
            "${entry.key.name}=$mae"
        }.joinToString { it }
    }

    companion object {
        private const val TAG = "OCRService"
    }
}
