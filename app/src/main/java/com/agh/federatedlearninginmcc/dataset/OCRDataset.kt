package com.agh.federatedlearninginmcc.dataset

import com.agh.federatedlearninginmcc.ml.DataUtils
import com.agh.federatedlearninginmcc.ml.ModelVariant
import com.agh.federatedlearninginmcc.ml.NormalizationStats
import com.agh.federatedlearninginmcc.ocr.BenchmarkInfo
import com.agh.federatedlearninginmcc.ocr.ImageInfo
import java.time.Instant
import java.time.ZoneId
import kotlin.time.Duration

abstract class OCRDataset {
    // TODO make all add/get thread safe
    private val datasetObservers: MutableList<(modelVariant: ModelVariant) -> Unit> = mutableListOf()

    abstract fun addLocallyComputedTimeSample(imgInfo: ImageInfo, computationTime: Duration)
    abstract fun addCloudComputedTimeSample(
        imgInfo: ImageInfo, computationTime: Duration, transmissionTime: Duration, executedAt: Instant)
    abstract fun addBenchmarkInfo(benchmarkInfo: BenchmarkInfo)

    // full dataset for simplicity, should probably be in batches
    abstract fun getLocalTimeDataset(benchmarkInfo: BenchmarkInfo): Dataset
    abstract fun getCloudComputationTimeDataset(benchmarkInfo: BenchmarkInfo): Dataset
    abstract fun getCloudTransmissionTimeDataset(benchmarkInfo: BenchmarkInfo): Dataset

    abstract fun getBenchmarkInfo(): BenchmarkInfo?
    abstract fun getLocalTimeDatasetSize(): Int
    abstract fun getCloudComputationTimeDatasetSize(): Int
    abstract fun getCloudTransmissionTimeDatasetSize(): Int

    abstract fun clear()

    fun getDataset(modelVariant: ModelVariant, benchmarkInfo: BenchmarkInfo): Dataset {
        return when(modelVariant) {
            ModelVariant.LOCAL_TIME -> getLocalTimeDataset(benchmarkInfo)
            ModelVariant.CLOUD_COMPUTATION_TIME -> getCloudComputationTimeDataset(benchmarkInfo)
            ModelVariant.CLOUD_TRANSMISSION_TIME -> getCloudTransmissionTimeDataset(benchmarkInfo)
        }
    }

    fun getNormalizationStats(modelVariant: ModelVariant, benchmarkInfo: BenchmarkInfo): NormalizationStats {
        val (xs, ys) = getDataset(modelVariant, benchmarkInfo)
        // TODO cache this
        val normalizationStats = DataUtils.getNormalizationStats(xs, ys, modelVariant.modelConfig.inputDimensions)
        // TODO how to handle this properly, we can't standardize as all values are equal
        getBenchmarkInfoDims(modelVariant).forEach { dim ->
            normalizationStats.xMeans[dim] = 0.0f
            normalizationStats.xStds[dim] = 1.0f
        }
        return normalizationStats
    }

    fun getDatasetSize(modelVariant: ModelVariant): Int {
        return when(modelVariant) {
            ModelVariant.LOCAL_TIME -> getLocalTimeDatasetSize()
            ModelVariant.CLOUD_COMPUTATION_TIME -> getCloudComputationTimeDatasetSize()
            ModelVariant.CLOUD_TRANSMISSION_TIME -> getCloudTransmissionTimeDatasetSize()
        }
    }

    fun getBenchmarkInfoDims(modelVariant: ModelVariant): IntArray {
        return intArrayOf(0)
    }

    fun addDatasetUpdatedObserver(observer: (modelVariant: ModelVariant) -> Unit) {
        datasetObservers.add(observer)
    }

    protected fun notifyDatasetUpdated(modelVariant: ModelVariant) {
        datasetObservers.forEach { it(modelVariant) }
    }

    fun toTimeOfDay(instant: Instant): Float {
        val secondOfDay = instant.atZone(ZoneId.of("UTC")).toLocalTime().toSecondOfDay()
        return secondOfDay.toFloat() / (24 * 3600)
    }

    fun createLocalTimeXSample(benchmarkInfo: BenchmarkInfo, imgInfo: ImageInfo): FloatArray {
        val res = FloatArray(4)
        res[0] = benchmarkInfo.meanComputationTime.inWholeMilliseconds.toFloat()
        res[1] = imgInfo.width.toFloat()
        res[2] = imgInfo.height.toFloat()
        res[3] = imgInfo.sizeBytes.toFloat()
        return res
    }

    fun createCloudTimeXSample(benchmarkInfo: BenchmarkInfo, imgInfo: ImageInfo, timeOfDay: Float): FloatArray {
        val res = FloatArray(5)
        res[0] = benchmarkInfo.meanComputationTime.inWholeMilliseconds.toFloat()
        res[1] = imgInfo.width.toFloat()
        res[2] = imgInfo.height.toFloat()
        res[3] = imgInfo.sizeBytes.toFloat()
        res[4] = timeOfDay
        return res
    }
}
