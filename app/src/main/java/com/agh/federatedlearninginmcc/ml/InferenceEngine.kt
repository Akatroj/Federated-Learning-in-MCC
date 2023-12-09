package com.agh.federatedlearninginmcc.ml

import android.util.Log
import com.agh.federatedlearninginmcc.dataset.OCRDataset
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

        // TODO energy, cloud info...
        val localTime = localTimeModel.predict(ocrDataset.convertLocallyComputedTimeSample(imgInfo))
        val cloudTime = cloudTimeModel.predict(ocrDataset.convertCloudComputedTimeSample(imgInfo))

        val localCost = computeLocalCost(localTime)
        val cloudCost = computeCloudCost(cloudTime)
        Log.d(TAG, "Predicted local time: $localTime, cloud time $cloudCost; local cost: $localCost, cloud cost: $cloudCost")

        // TODO this should be randomized to some degree to give client a chance to get some data for both local and cloud cases
        return localCost < cloudCost
    }

    private fun computeCloudCost(cloudTime: Float): Float {
        return 1.5f * cloudTime
    }

    private fun computeLocalCost(localTime: Float): Float {
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
