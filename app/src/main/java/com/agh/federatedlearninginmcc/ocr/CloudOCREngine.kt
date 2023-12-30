package com.agh.federatedlearninginmcc.ocr

import java.io.File

interface CloudOCREngine {
    fun doOCR(img: File): CloudOCRInfo
}
