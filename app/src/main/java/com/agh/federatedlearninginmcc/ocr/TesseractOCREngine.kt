package com.agh.federatedlearninginmcc.ocr

import com.googlecode.tesseract.android.TessBaseAPI
import java.io.File

class TesseractOCREngine(
    tesseractDataPath: String,
    language: String = "eng",
    private val artificialDelayFactor: Float? = null
) : LocalOCREngine {
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
            val start = System.currentTimeMillis()
            tesseract.setImage(img)
            val res = tesseract.utF8Text
            val totalTime = System.currentTimeMillis() - start
            if (artificialDelayFactor != null) {
                Thread.sleep((artificialDelayFactor * totalTime).toLong())
            }
            return res
        }
    }
}
