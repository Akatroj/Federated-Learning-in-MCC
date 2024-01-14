package com.agh.federatedlearninginmcc.ml

import android.content.Context
import org.tensorflow.lite.support.common.FileUtil
import java.io.File

object ModelFactory {
    fun createModel(context: Context, variant: ModelVariant, restoreFromFile: Boolean): FlowerRegressionModel {
        val model = FlowerRegressionModel(
            FileUtil.loadMappedFile(context, variant.modelConfig.modelAssetsFile),
            variant.modelConfig.inputDimensions,
            variant.modelConfig.name,
            variant.modelConfig.layerSizes,
        )
        if (restoreFromFile) {
            model.restoreFromDisk(File(context.filesDir, variant.modelConfig.modelTrainedFile).absolutePath)
        }
        return model
    }
}

