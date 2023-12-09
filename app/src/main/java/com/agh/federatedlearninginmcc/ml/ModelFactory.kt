package com.agh.federatedlearninginmcc.ml

import android.content.Context
import org.tensorflow.lite.support.common.FileUtil

object ModelFactory {
    fun createModel(context: Context, variant: ModelVariant, normalizationStats: NormalizationStats) =
        FlowerRegressionModel(
            FileUtil.loadMappedFile(context, variant.modelConfig.modelFile),
            variant.modelConfig.inputDimensions,
            variant.modelConfig.name,
            variant.modelConfig.layerSizes,
            normalizationStats.xMeans,
            normalizationStats.xStds,
            normalizationStats.yMean,
            normalizationStats.yStd
        )
}

