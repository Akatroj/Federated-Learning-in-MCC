package com.agh.federatedlearninginmcc.ocr

import com.agh.federatedlearninginmcc.InferenceEngine
import java.io.File

class OCRService(
    private val localOCREngine: LocalOCREngine,
    private val cloudOCREngine: CloudOCREngine,
    private val inferenceEngine: InferenceEngine
){
    fun doOCR(img: File): String {
        return if (inferenceEngine.shouldRunOCRLocally()) {
            dispatchLocalOCR(img)
        } else {
            dispatchCloudOCR(img)
        }
    }

    private fun dispatchLocalOCR(img: File): String {
        return localOCREngine.doOCR(img)
    }

    private fun dispatchCloudOCR(img: File): String {
        return cloudOCREngine.doOCR(img).text
    }
}