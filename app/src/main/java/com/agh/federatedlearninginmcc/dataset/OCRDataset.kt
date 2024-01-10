package com.agh.federatedlearninginmcc.dataset

import com.agh.federatedlearninginmcc.ml.DataUtils
import com.agh.federatedlearninginmcc.ml.ModelVariant
import com.agh.federatedlearninginmcc.ml.NormalizationStats
import com.agh.federatedlearninginmcc.ocr.BenchmarkInfo
import com.agh.federatedlearninginmcc.ocr.ImageInfo
import java.time.Instant
import java.time.ZoneId
import kotlin.time.Duration

const val BENCHMARK_TIME_SCALE_COEF = 2000.0f

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

    private fun getBenchmarkInfoDims(modelVariant: ModelVariant): IntArray {
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
        val res = FloatArray(6)
        assert(res.size == ModelVariant.LOCAL_TIME.modelConfig.inputDimensions)
        res[0] = benchmarkInfo.meanComputationTime.inWholeMilliseconds.toFloat() / BENCHMARK_TIME_SCALE_COEF
        fillImageInfo(imgInfo, res, 1)
        return res
    }

    fun createCloudTimeXSample(benchmarkInfo: BenchmarkInfo, imgInfo: ImageInfo, timeOfDay: Float): FloatArray {
        val res = FloatArray(7)
        assert(res.size == ModelVariant.CLOUD_COMPUTATION_TIME.modelConfig.inputDimensions)
        res[0] = benchmarkInfo.meanComputationTime.inWholeMilliseconds.toFloat() / BENCHMARK_TIME_SCALE_COEF
        fillImageInfo(imgInfo, res, 1)
        res[6] = timeOfDay
        return res
    }

    private fun fillImageInfo(imgInfo: ImageInfo, arr: FloatArray, start: Int) {
        arr[start] = imgInfo.width.toFloat()
        arr[start + 1] = imgInfo.height.toFloat()
        arr[start + 2] = imgInfo.sizeBytes.toFloat()
        arr[start + 3] = imgInfo.textToBackgroundRatio
        arr[start + 4] = imgInfo.numTextLines.toFloat()
    }
}
