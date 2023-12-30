package com.agh.federatedlearninginmcc.ocr

import kotlin.time.Duration

data class CloudOCRInfo(
    val text: String,
    val computationTime: Duration
)
