package com.agh.federatedlearninginmcc.ocr

import android.util.Log
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

import okhttp3.OkHttpClient
import okhttp3.Request

class TransmissionTester(private val infoUrl: String) {
    private val client = OkHttpClient()

    companion object {
        const val TAG = "TransmissionTester"
    }

    // blocking
    fun runTransmissionTest(): TransmissionTestInfo {
        val request = Request.Builder()
            .url(infoUrl)
            .get()
            .build()

        val start = System.currentTimeMillis()

        client.newCall(request).execute().use { response ->
            val rtt = System.currentTimeMillis() - start
            Log.d(TAG, "Got info respose")
            if (!response.isSuccessful || response.body == null)
                Log.e(TAG, "Failed to call Info service")
            val clusterInfo = Json.decodeFromString<ClusterInfo>(response.body!!.string())

            val testInfo = TransmissionTestInfo(clusterInfo.nodeNumber, rtt.toInt())
            Log.d(TAG, testInfo.toString())
            return testInfo
        }
    }
}

@Serializable
data class ClusterInfo(val nodeNumber: Int)

data class TransmissionTestInfo(
    val nodes: Int,
    val rttMs: Int,
)
