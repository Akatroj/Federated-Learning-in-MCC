package com.agh.federatedlearninginmcc.ml

import android.util.Log
import com.agh.federatedlearninginmcc.dataset.OCRDataset
import com.agh.federatedlearninginmcc.ocr.BenchmarkInfo
import com.agh.federatedlearninginmcc.ocr.ImageInfo
import java.time.Instant

class InferenceEngine(
    private val ocrDataset: OCRDataset,
    private val localTimeModel: FlowerRegressionModel,
    private val cloudComputationTimeModel: FlowerRegressionModel,
    private val cloudTransmissionTimeModel: FlowerRegressionModel,
    private val runningLocation: RunningLocation = RunningLocation.PREDICT
) {
    fun shouldRunOCRLocally(imgInfo: ImageInfo): Boolean {
        if (runningLocation != RunningLocation.PREDICT)
            return runningLocation == RunningLocation.FORCE_LOCAL

        val benchmarkInfo = ocrDataset.getBenchmarkInfo() ?: return false

        val localCost = computeLocalCost(imgInfo, benchmarkInfo)
        val cloudCost = computeCloudCost(imgInfo, benchmarkInfo)
        Log.d(TAG, "local cost: $localCost, cloud cost: $cloudCost")

        // TODO this should be randomized to some degree to give client a chance to get some data for both local and cloud cases
        return localCost < cloudCost
    }

    private fun computeCloudCost(imgInfo: ImageInfo, benchmarkInfo: BenchmarkInfo): Float {
        val timeOfDay = ocrDataset.toTimeOfDay(Instant.now())
        val cloudTimeX = ocrDataset.createCloudTimeXSample(benchmarkInfo, imgInfo, timeOfDay)

        val cloudComputationTime = normalizeAndPredict(
            cloudTimeX, ModelVariant.CLOUD_COMPUTATION_TIME, cloudComputationTimeModel, benchmarkInfo)
        val cloudTransmissionTime = normalizeAndPredict(
            cloudTimeX, ModelVariant.CLOUD_TRANSMISSION_TIME, cloudTransmissionTimeModel, benchmarkInfo)

        Log.d(TAG, "Predicted cloud computation time: $cloudComputationTime, cloud transmission time: $cloudTransmissionTime")

        return 1.5f * (cloudComputationTime + cloudTransmissionTime)
    }

    private fun computeLocalCost(imgInfo: ImageInfo, benchmarkInfo: BenchmarkInfo): Float {
        val localTime = normalizeAndPredict(
            ocrDataset.createLocalTimeXSample(benchmarkInfo, imgInfo),
            ModelVariant.LOCAL_TIME,
            localTimeModel,
            benchmarkInfo
        )
        Log.d(TAG, "Predicted local time: $localTime")

        return localTime
    }

    private fun normalizeAndPredict(x: FloatArray, modelVariant: ModelVariant, model: FlowerRegressionModel, benchmarkInfo: BenchmarkInfo): Float {
        val normalizationStats = ocrDataset.getNormalizationStats(modelVariant, benchmarkInfo)
        val pred = model.predict(
            DataUtils.normalize(listOf(x), normalizationStats.xMeans, normalizationStats.xStds)[0]
        )
        return pred * normalizationStats.yStd + normalizationStats.yMean
    }

    companion object {
        private const val TAG = "InferenceEngine"
    }
}

enum class RunningLocation {
    PREDICT,
    FORCE_LOCAL,
    FORCE_CLOUD
}
