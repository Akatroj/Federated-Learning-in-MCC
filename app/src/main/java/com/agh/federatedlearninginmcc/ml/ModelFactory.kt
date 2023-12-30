package com.agh.federatedlearninginmcc.ml

import android.content.Context
import org.tensorflow.lite.support.common.FileUtil

object ModelFactory {
    fun createModel(context: Context, variant: ModelVariant) =
        FlowerRegressionModel(
            FileUtil.loadMappedFile(context, variant.modelConfig.modelAssetsFile),
            variant.modelConfig.inputDimensions,
            variant.modelConfig.name,
            variant.modelConfig.layerSizes,
        )
}
