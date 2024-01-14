package com.agh.federatedlearninginmcc.ocr

import android.graphics.BitmapFactory
import java.io.File

data class ImageInfo(
    val width: Int,
    val height: Int,
    val sizeBytes: Int,
    val textToBackgroundRatio: Float,
    val numTextLines: Int,
)

object ImageUtils {
    private const val TEXT_THRESHOLD = 128
    private const val MIN_TEXT_PIXELS_IN_LINE_COEF = 150
    private const val MIN_LINE_HEIGHT_COEF = 150
    private const val MIN_LINE_PADDING = 3

    fun createImageInfo(img: File): ImageInfo {
        val bitmap = BitmapFactory.decodeFile(img.path, BitmapFactory.Options())
        val imgBuff = IntArray(bitmap.width * bitmap.height) // TODO maybe don't realloc imgBuff everytime
        bitmap.getPixels(imgBuff, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        val minNumTextPixels = (bitmap.width / MIN_TEXT_PIXELS_IN_LINE_COEF).coerceAtLeast(3)
        val minLineHeight = (bitmap.height / MIN_LINE_HEIGHT_COEF).coerceAtLeast(2)

        var totalTextPixels = 0
        var lines = 0
        var previousLineEnd = -1
        var lineStart = -1

        for (i in 0..<bitmap.height) {
            val rowTextPixels = countTextPixelsInRow(imgBuff, i, bitmap.width)
            totalTextPixels += rowTextPixels
            if (rowTextPixels > minNumTextPixels) {
                if (lineStart == -1) {
                    lineStart = i
                    if (previousLineEnd != -1 && (i - previousLineEnd < MIN_LINE_PADDING)) {
                        lines--
                    }
                }
            } else if (lineStart != -1) {
                if (i - lineStart > minLineHeight) {
                    lines++
                    previousLineEnd = i
                }
                lineStart = -1
            }
        }

        if (lineStart != -1 && (bitmap.height - lineStart > minLineHeight)) {
            lines++
        }

        return ImageInfo(
            width = bitmap.width,
            height = bitmap.height,
            sizeBytes = img.length().toInt(),
            textToBackgroundRatio = totalTextPixels / (bitmap.width * bitmap.height).toFloat(),
            numTextLines = lines,
        )
    }

    private fun countTextPixelsInRow(imgBuff: IntArray, row: Int, width: Int): Int {
        var res = 0
        val start = row * width
        for (j in 0..<width) {
            if (convertToGrayscale(imgBuff[start + j]) < TEXT_THRESHOLD) {
                res++
            }
        }
        return res
    }

    private fun convertToGrayscale(color: Int): Int {
        return ((color shr 16 and 0xFF) * 0.299f + (color shr 8 and 0xFF) * 0.587f + (color and 0xFF) * 0.114f).toInt()
    }
}
