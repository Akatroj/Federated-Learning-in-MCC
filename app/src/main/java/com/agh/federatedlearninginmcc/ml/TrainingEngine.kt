package com.agh.federatedlearninginmcc.ml

import android.content.Context
import android.util.Log
import com.agh.federatedlearninginmcc.dataset.OCRDataset
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.selects.select

class TrainingEngine(
    private val context: Context,
    private val serverIP: String,
    private val dataset: OCRDataset,
    private val minSamplesToJoinTraining: Int = 10,
    private val minNewSamplesToUpdateDatasets: Int = 10
) {
    private val trainers: MutableMap<ModelVariantKey, FlowerRegressionTrainer?> = mutableMapOf()
    private val usedDatasetSizes: MutableMap<ModelVariantKey, Int> = mutableMapOf()
    private val trainerCreatedChannel: Channel<ModelVariant> = Channel(Channel.UNLIMITED)

    init {
        ModelVariant.entries.forEach { modelVariant ->
            val datasetSize = dataset.getDatasetSize(modelVariant)
            if (datasetSize >= minSamplesToJoinTraining)  {
                trainers[modelVariant.key] = initTrainer(modelVariant)
                runBlocking { trainerCreatedChannel.send(modelVariant) }
            }
            usedDatasetSizes[modelVariant.key] = datasetSize
        }

        dataset.addDatasetUpdatedObserver { modelVariant ->
            val datasetSize = dataset.getDatasetSize(modelVariant)
            if (trainers[modelVariant.key] == null) {
                if (datasetSize >= minNewSamplesToUpdateDatasets) {
                    trainers[modelVariant.key] = initTrainer(modelVariant)
                    runBlocking { trainerCreatedChannel.send(modelVariant) }
                }
            } else if (datasetSize - usedDatasetSizes[modelVariant.key]!! >= minNewSamplesToUpdateDatasets) {
                // TODO reset trainer dataset
            }
        }
    }

    suspend fun joinFederatedTraining() = coroutineScope {
        while (true) {
            select {
                trainerCreatedChannel.onReceive { modelVariant ->
                    launch {
                        Log.d(modelVariant.trainerConfig.tag, "Joining federated training")
                        createFlowerRegressionService(
                            serverIP,
                            modelVariant.trainerConfig.port,
                            false,
                            trainers[modelVariant.key]!!) { Log.d(modelVariant.trainerConfig.tag, it)}
                    }
                }
            }
        }
    }

    private fun initTrainer(modelVariant: ModelVariant): FlowerRegressionTrainer {
        // TODO it's a pretty strong assumption that mean+std of samples from this client matches
        // those of global dataset, we should get them from server (and send them for averaging too)
        val (xs, ys) = dataset.getDataset(modelVariant)
        val normalizationStats = DataUtils.getNormalizationStats(xs, ys, modelVariant.modelConfig.inputDimensions)

        val model = ModelFactory.createModel(context, modelVariant, normalizationStats)
        val (normalizedXs, normalizedYs) = normalizeData(xs, ys, normalizationStats)

        return FlowerRegressionTrainer(model, normalizedXs, normalizedYs)
    }

    private fun normalizeData(
        xs: List<FloatArray>,
        ys: FloatArray,
        normalizationStats: NormalizationStats
    ): Pair<List<FloatArray>, FloatArray> {
        return Pair(
            DataUtils.normalize(xs, normalizationStats.xMeans, normalizationStats.xStds),
            DataUtils.normalize(ys, normalizationStats.yMean, normalizationStats.yStd)
        )
    }
}
