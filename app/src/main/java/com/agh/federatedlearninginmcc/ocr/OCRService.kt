package com.agh.federatedlearninginmcc.ocr

import android.graphics.BitmapFactory
import android.util.Log
import com.agh.federatedlearninginmcc.ml.InferenceEngine
import com.agh.federatedlearninginmcc.dataset.OCRDataset
import java.io.File
import kotlin.time.DurationUnit
import kotlin.time.toDuration

class OCRService(
    private val localOCREngine: LocalOCREngine,
    private val cloudOCREngine: CloudOCREngine,
    private val inferenceEngine: InferenceEngine,
    private val ocrDataset: OCRDataset
){
    fun doOCR(img: File): String {
        val imgInfo = getImageInfo(img)
        val start = System.currentTimeMillis()
        // TODO measure energy somehow

        return if (inferenceEngine.shouldRunOCRLocally(imgInfo)) {
            Log.d(TAG, "running OCR locally")
            val ocrResult = localOCREngine.doOCR(img)
            val ocrTime = System.currentTimeMillis() - start
            Log.d(TAG, "OCR local time: $ocrTime")
            ocrDataset.addLocallyComputedTimeSample(imgInfo, ocrTime.toDuration(DurationUnit.MILLISECONDS))
            ocrResult
        } else {
            Log.d(TAG, "running OCR remotely")
            val ocrResult = cloudOCREngine.doOCR(img)
            val ocrTime = System.currentTimeMillis() - start
            Log.d(TAG, "OCR remote time: $ocrTime")
            ocrDataset.addCloudComputedTimeSample(imgInfo, ocrTime.toDuration(DurationUnit.MILLISECONDS))
            ocrResult.text
        }
    }

    private fun getImageInfo(img: File): ImageInfo {
        val bitmapOptions = BitmapFactory.Options()
        bitmapOptions.inJustDecodeBounds = true
        BitmapFactory.decodeFile(img.path, bitmapOptions)
        return ImageInfo(
            width = bitmapOptions.outWidth,
            height = bitmapOptions.outHeight,
            sizeBytes = img.length().toInt()
        )
    }

    companion object {
        private const val TAG = "OCRService"
    }
}
