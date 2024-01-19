package com.agh.federatedlearninginmcc.ocr

import com.agh.federatedlearninginmcc.ml.Inference

data class OCRResultData(
    val result: String,
    val prediction: Inference,
    val actualTimeMs: Float
)