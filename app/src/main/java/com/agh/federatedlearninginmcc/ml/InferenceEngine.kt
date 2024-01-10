package com.agh.federatedlearninginmcc.ml

import com.agh.federatedlearninginmcc.dataset.OCRDataset
import com.agh.federatedlearninginmcc.ocr.BenchmarkInfo
import com.agh.federatedlearninginmcc.ocr.ImageInfo
import java.time.Instant

class InferenceEngine(
    private val ocrDataset: OCRDataset,
    private val localTimeModel: FlowerRegressionModel,
    private val cloudComputationTimeModel: FlowerRegressionModel,
    private val cloudTransmissionTimeModel: FlowerRegressionModel,
    private val runningLocation: RunningLocation = RunningLocation.PREDICT,
    private val explorationChance: Double = .0
) {
    fun predictComputationTimes(imgInfo: ImageInfo): Inference {
        if (runningLocation != RunningLocation.PREDICT)
            return Inference.Forced(runningLocation == RunningLocation.FORCE_LOCAL)

        val benchmarkInfo = ocrDataset.getBenchmarkInfo() ?: return Inference.Forced(false)

        val localInference = computeLocalCost(imgInfo, benchmarkInfo)
        val cloudInference = computeCloudCost(imgInfo, benchmarkInfo)

        val runLocally = if (Math.random() < explorationChance) {
            // with some probability take random option in order to collect more data for that case
            // exploration chance should be based on the dataset size
            Math.random() > 0.5
        } else {
            localInference.cost < cloudInference.cost
        }
        return Inference(runLocally, localInference, cloudInference)
    }

    private fun computeCloudCost(imgInfo: ImageInfo, benchmarkInfo: BenchmarkInfo): CloudInference {
        val timeOfDay = ocrDataset.toTimeOfDay(Instant.now())
        val cloudTimeX = ocrDataset.createCloudTimeXSample(benchmarkInfo, imgInfo, timeOfDay)

        val cloudComputationTime = normalizeAndPredict(
            cloudTimeX, ModelVariant.CLOUD_COMPUTATION_TIME, cloudComputationTimeModel, benchmarkInfo)
        val cloudTransmissionTime = normalizeAndPredict(
            cloudTimeX, ModelVariant.CLOUD_TRANSMISSION_TIME, cloudTransmissionTimeModel, benchmarkInfo)

        val totalCost = 1.5f * (cloudComputationTime + cloudTransmissionTime)
        return CloudInference(totalCost, cloudComputationTime, cloudTransmissionTime)
    }

    private fun computeLocalCost(imgInfo: ImageInfo, benchmarkInfo: BenchmarkInfo): TimeInference {
        val localTime = normalizeAndPredict(
            ocrDataset.createLocalTimeXSample(benchmarkInfo, imgInfo),
            ModelVariant.LOCAL_TIME,
            localTimeModel,
            benchmarkInfo
        )
        val localCost = 1.0f * localTime

        return TimeInference(localCost, localTime)
    }

    private fun normalizeAndPredict(x: FloatArray, modelVariant: ModelVariant, model: FlowerRegressionModel, benchmarkInfo: BenchmarkInfo): Float {
        val normalizationStats = ocrDataset.getNormalizationStats(modelVariant, benchmarkInfo)
        val pred = model.predict(
            DataUtils.normalize(listOf(x), normalizationStats.xMeans, normalizationStats.xStds)[0]
        )
        return pred * normalizationStats.yStd + normalizationStats.yMean
    }
}

data class TimeInference(
    val cost: Float,
    val timeMs: Float
) {
    companion object {
        fun Forced(): TimeInference {
            return TimeInference(.0f, .0f)
        }
    }
}

data class CloudInference(
    val cost: Float,
    val computationTimeMs: Float,
    val transmissionTimeMs: Float
) {
    companion object {
        fun Forced(): CloudInference {
            return CloudInference(.0f, .0f, .0f)
        }
    }
}

data class Inference(
    val shouldRunLocally: Boolean,
    val localTime: TimeInference,
    val cloudTime: CloudInference
) {
    companion object {
        fun Forced(shouldRunLocally: Boolean): Inference {
            return Inference(shouldRunLocally, TimeInference.Forced(), CloudInference.Forced())
        }
    }
}

enum class RunningLocation {
    PREDICT,
    FORCE_LOCAL,
    FORCE_CLOUD
}
