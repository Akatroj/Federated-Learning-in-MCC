package com.agh.federatedlearninginmcc.ml

import android.util.Log
import com.agh.federatedlearninginmcc.dataset.OCRDataset
import com.agh.federatedlearninginmcc.ocr.BenchmarkHandler
import com.agh.federatedlearninginmcc.ocr.BenchmarkInfo
import com.agh.federatedlearninginmcc.ocr.ImageInfo

class InferenceEngine(
    private val ocrDataset: OCRDataset,
    private val localTimeModel: FlowerRegressionModel,
    private val cloudTimeModel: FlowerRegressionModel,
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
        val cloudTimeNormalizationStats = ocrDataset.getNormalizationStats(ModelVariant.CLOUD_TIME, benchmarkInfo)

        val cloudTime = cloudTimeModel.predict(
            DataUtils.normalize(
                listOf(ocrDataset.createCloudTimeXSample(benchmarkInfo, imgInfo)),
                cloudTimeNormalizationStats.xMeans, cloudTimeNormalizationStats.xStds)[0]
        ) * cloudTimeNormalizationStats.yStd + cloudTimeNormalizationStats.yMean

        Log.d(TAG, "Predicted cloud time: $cloudTime")

        return 1.5f * cloudTime
    }

    private fun computeLocalCost(imgInfo: ImageInfo, benchmarkInfo: BenchmarkInfo): Float {
        val localTimeNormalizationStats = ocrDataset.getNormalizationStats(ModelVariant.LOCAL_TIME, benchmarkInfo)
        val localTime = localTimeModel.predict(
            DataUtils.normalize(
                listOf(ocrDataset.createLocalTimeXSample(benchmarkInfo, imgInfo)),
                localTimeNormalizationStats.xMeans, localTimeNormalizationStats.xStds)[0]
        ) * localTimeNormalizationStats.yStd + localTimeNormalizationStats.yMean

        Log.d(TAG, "Predicted local time: $localTime")

        return localTime
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
