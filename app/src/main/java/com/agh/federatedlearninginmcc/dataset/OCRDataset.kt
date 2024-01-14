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
const val NODES_NUM_SCALE_COEF = 5.0f
const val RTT_SCALE_COEF = 1000.0f

abstract class OCRDataset {
    // TODO make all add/get thread safe
    private val datasetObservers: MutableList<(modelVariant: ModelVariant) -> Unit> = mutableListOf()

    abstract fun addLocallyComputedTimeSample(imgInfo: ImageInfo, computationTime: Duration)
    abstract fun addCloudComputedTimeSample(
        imgInfo: ImageInfo,
        computationTime: Duration,
        transmissionTime: Duration,
        executedAt: Instant,
        numNodes: Int,
        rttMillis: Int
    )
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
        modelVariant.modelConfig.dontStandardizeDims.forEach { dim ->
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
        assert(ModelVariant.LOCAL_TIME.modelConfig.dontStandardizeDims.contains(0))
        res[0] = benchmarkInfo.meanComputationTime.inWholeMilliseconds.toFloat() / BENCHMARK_TIME_SCALE_COEF
        fillImageInfo(imgInfo, res, 1)
        return res
    }

    fun createCloudComputationTimeXSample(benchmarkInfo: BenchmarkInfo, imgInfo: ImageInfo, numNodes: Int, timeOfDay: Float): FloatArray {
        val res = FloatArray(8)
        assert(res.size == ModelVariant.CLOUD_COMPUTATION_TIME.modelConfig.inputDimensions)
        assert(ModelVariant.CLOUD_COMPUTATION_TIME.modelConfig.dontStandardizeDims.containsAll(listOf(0, 6, 7)))
        res[0] = benchmarkInfo.meanComputationTime.inWholeMilliseconds.toFloat() / BENCHMARK_TIME_SCALE_COEF
        fillImageInfo(imgInfo, res, 1)
        res[6] = timeOfDay
        res[7] = numNodes.toFloat() / NODES_NUM_SCALE_COEF
        return res
    }

    fun createCloudTransmissionTimeXSample(benchmarkInfo: BenchmarkInfo, imgInfo: ImageInfo, numNodes: Int, rttMillis: Int, timeOfDay: Float): FloatArray {
        val res = FloatArray(5)
        assert(res.size == ModelVariant.CLOUD_TRANSMISSION_TIME.modelConfig.inputDimensions)
        assert(ModelVariant.CLOUD_TRANSMISSION_TIME.modelConfig.dontStandardizeDims.containsAll(listOf(0, 2, 3, 4)))
        res[0] = benchmarkInfo.meanComputationTime.inWholeMilliseconds.toFloat() / BENCHMARK_TIME_SCALE_COEF
        res[1] = imgInfo.sizeBytes.toFloat()
        res[2] = timeOfDay
        res[3] = numNodes.toFloat() / NODES_NUM_SCALE_COEF
        res[4] = rttMillis.toFloat() / RTT_SCALE_COEF
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
