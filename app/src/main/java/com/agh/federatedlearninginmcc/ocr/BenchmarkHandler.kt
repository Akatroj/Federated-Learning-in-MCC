package com.agh.federatedlearninginmcc.ocr

import android.util.Log
import com.agh.federatedlearninginmcc.dataset.OCRDataset
import java.io.File
import kotlin.time.Duration
import kotlin.time.DurationUnit
import kotlin.time.toDuration

class BenchmarkHandler(
    private val dataset: OCRDataset,
    private val localOCREngine: LocalOCREngine,
    private val benchmarkImageFile: File,
) {
    private val benchmarkLock = Any()

    fun assertHasRunBenchmark() {
        synchronized(benchmarkLock) {
            if (dataset.getBenchmarkInfo() == null) {
                Log.d(TAG, "Running benchmark")
                dataset.addBenchmarkInfo(runBenchmark())
            }
        }
    }

    private fun runBenchmark(repeats: Int = 3): BenchmarkInfo {
        val start = System.currentTimeMillis()
        for (i in 0..<repeats) {
            localOCREngine.doOCR(benchmarkImageFile)
        }
        val meanComputationTime = (System.currentTimeMillis() - start) / repeats
        return BenchmarkInfo(repeats, meanComputationTime.toDuration(DurationUnit.MILLISECONDS))
    }

    companion object {
        private const val TAG = "BenchmarkHandler"
    }
}

data class BenchmarkInfo(
    val repeats: Int,
    val meanComputationTime: Duration,
)
