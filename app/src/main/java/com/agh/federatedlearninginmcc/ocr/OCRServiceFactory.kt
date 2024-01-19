package com.agh.federatedlearninginmcc.ocr

import android.content.Context
import com.agh.federatedlearninginmcc.TrainingFragment
import com.agh.federatedlearninginmcc.dataset.OCRDataset
import com.agh.federatedlearninginmcc.ml.InferenceEngine
import com.agh.federatedlearninginmcc.ml.ModelFactory
import com.agh.federatedlearninginmcc.ml.ModelVariant
import com.agh.federatedlearninginmcc.ml.RunningLocation
import java.io.File

object Config {
    //    const val K8S_OCR_URL = "http://172.18.0.3:31555/base64"
    const val K8S_OCR_URL = "http://34.116.247.243:8080/base64"

    //    const val FLOWER_SERVER_IP = "10.0.2.2" // localhost from emulator
    const val FLOWER_SERVER_IP = "34.116.231.98"
    const val TRANSMISSION_TESTING_URL = "http://34.107.121.153:8080/info"
    const val TOTAL_IMAGES = 600
    const val DELAY_FACTOR = 1.0f // totalTime = originalTime + delayFactor * originalTime
}

object OCRServiceFactory {
    fun create(
        context: Context,
        dataset: OCRDataset,
        runningLocation: RunningLocation = RunningLocation.PREDICT,
        restoreModels: Boolean = true,
        saveNewSamples: Boolean = true
    ): OCRService {
        return create(
            context,
            dataset,
            TransmissionTester(Config.TRANSMISSION_TESTING_URL).runTransmissionTest(),
            runningLocation,
            restoreModels,
            saveNewSamples
        )
    }

    fun create(
        context: Context,
        dataset: OCRDataset,
        transmissionTestInfo: TransmissionTestInfo,
        runningLocation: RunningLocation = RunningLocation.PREDICT,
        restoreModels: Boolean = true,
        saveNewSamples: Boolean = true
    ): OCRService {
        val (localModel, cloudComputationModel, cloudTransmissionModel) = listOf(
            ModelVariant.LOCAL_TIME,
            ModelVariant.CLOUD_COMPUTATION_TIME,
            ModelVariant.CLOUD_TRANSMISSION_TIME
        ).map { ModelFactory.createModel(context, it, restoreModels) }

        return OCRService(
            TesseractOCREngineFactory.create(context),
            K8sOCREngine(Config.K8S_OCR_URL),
            InferenceEngine(
                dataset,
                localModel,
                cloudComputationModel,
                cloudTransmissionModel,
                runningLocation
            ),
            dataset,
            transmissionTestInfo,
            saveNewSamples = saveNewSamples
        )
    }
}

object TesseractOCREngineFactory {
    fun create(context: Context): TesseractOCREngine {
        val tessDataDir = File(context.filesDir, "tessdata")
        if (!tessDataDir.exists()) {
            val tesseractTrainData = File(tessDataDir, "eng.traineddata")
            tessDataDir.mkdir()
            context.assets.open(tesseractTrainData.name).use { inputStream ->
                tesseractTrainData.outputStream().use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
        }
        return TesseractOCREngine(
            context.filesDir.path,
            artificialDelayFactor = Config.DELAY_FACTOR
        )
    }
}