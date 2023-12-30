package com.agh.federatedlearninginmcc.ocr

import com.googlecode.tesseract.android.TessBaseAPI
import java.io.File

class TesseractOCREngine(tesseractDataPath: String, language: String = "eng") : LocalOCREngine {
    private val tesseract: TessBaseAPI = TessBaseAPI()
    private val tesseractLock = Any()

    init {
        if (!tesseract.init(tesseractDataPath, language)) {
            tesseract.recycle()
            throw Exception("Failed to init tesseract")
        }
    }

    override fun doOCR(img: File): String {
        synchronized(tesseractLock) {
            tesseract.setImage(img)
            return tesseract.utF8Text
        }
    }
}
