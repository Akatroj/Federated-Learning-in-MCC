package com.agh.federatedlearninginmcc.ocr

import android.util.Log
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.io.IOException
import kotlin.io.encoding.Base64
import kotlin.io.encoding.ExperimentalEncodingApi

class K8sOCREngine(private val ocrUrl: String) : CloudOCREngine {
    private val client = OkHttpClient()

    @OptIn(ExperimentalEncodingApi::class)
    override fun doOCR(img: File): CloudOCRInfo {
        Log.d(TAG, "Sending OCR request")
        val ocrRequest = OCRRequest(Base64.encode(img.readBytes()))
        val body = Json.encodeToString(ocrRequest).toRequestBody("application/json".toMediaType())
        val request = Request.Builder()
            .url(ocrUrl)
            .post(body)
            .build()

        client.newCall(request).execute().use { response ->
            Log.d(TAG, "Got OCR response")
            if (!response.isSuccessful || response.body == null)
                throw IOException("Failed to call OCR cloud service")
            val ocrResponse = Json.decodeFromString<OCRResponse>(response.body!!.string())
            return@doOCR CloudOCRInfo(ocrResponse.result)
        }
    }

    companion object {
        private const val TAG = "K8sOCR"
    }
}

@Serializable
data class OCRRequest(val base64: String)

@Serializable
data class OCRResponse(val result: String, val version: String)
