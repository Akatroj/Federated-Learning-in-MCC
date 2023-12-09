package com.agh.federatedlearninginmcc.dataset

import com.agh.federatedlearninginmcc.ml.ModelVariant
import com.agh.federatedlearninginmcc.ocr.ImageInfo
import kotlin.time.Duration

abstract class OCRDataset {
    // TODO make all add/get thread safe
    private val datasetObservers: MutableList<(modelVariant: ModelVariant) -> Unit> = mutableListOf()

    abstract fun addLocallyComputedTimeSample(imgInfo: ImageInfo, computationTime: Duration)
    abstract fun addCloudComputedTimeSample(imgInfo: ImageInfo, totalComputationTime: Duration)

    // full dataset for simplicity, should probably be in batches
    abstract fun getLocalTimeDataset(): Pair<List<FloatArray>, FloatArray>
    abstract fun getCloudTimeDataset(): Pair<List<FloatArray>, FloatArray>

    abstract fun clear()

    fun getDataset(modelVariant: ModelVariant): Pair<List<FloatArray>, FloatArray> {
        return when(modelVariant) {
            ModelVariant.LOCAL_TIME -> getLocalTimeDataset()
            ModelVariant.CLOUD_TIME -> getCloudTimeDataset()
        }
    }

    fun getDatasetSize(modelVariant: ModelVariant): Int {
        return getDataset(modelVariant).first.size
    }

    fun addDatasetUpdatedObserver(observer: (modelVariant: ModelVariant) -> Unit) {
        datasetObservers.add(observer)
    }

    protected fun notifyDatasetUpdated(modelVariant: ModelVariant) {
        datasetObservers.forEach { it(modelVariant) }
    }

    fun convertLocallyComputedTimeSample(imgInfo: ImageInfo): FloatArray {
        val res = FloatArray(3)
        res[0] = imgInfo.width.toFloat()
        res[1] = imgInfo.height.toFloat()
        res[2] = imgInfo.sizeBytes.toFloat()
        return res
    }

    fun convertCloudComputedTimeSample(imgInfo: ImageInfo): FloatArray {
        return convertLocallyComputedTimeSample(imgInfo)
    }
}
