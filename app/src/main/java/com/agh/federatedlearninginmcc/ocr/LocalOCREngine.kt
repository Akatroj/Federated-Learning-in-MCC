package com.agh.federatedlearninginmcc.ocr

import java.io.File

interface LocalOCREngine {
    fun doOCR(img: File): String
}