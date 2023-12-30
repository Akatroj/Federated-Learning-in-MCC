package com.agh.federatedlearninginmcc.ml

import kotlin.math.sqrt

object DataUtils {
    fun mean(arr: FloatArray): Float {
        if (arr.isEmpty())
            return .0f
        return arr.average().toFloat()
    }

    fun mean(samples: List<FloatArray>, numFeatures: Int): FloatArray {
        val res = FloatArray(numFeatures)
        for (i in 0..<numFeatures) {
            res[i] = mean(samples[i])
        }
        return res
    }

    fun std(arr: FloatArray, mean: Float): Float {
        if (arr.isEmpty())
            return 1.0f
        return sqrt(arr.fold(.0f) {acc, cur -> acc + (cur - mean) * (cur - mean)})
    }

    fun std(samples: List<FloatArray>, means: FloatArray): FloatArray {
        val res = FloatArray(means.size)
        for (i in res.indices) {
            res[i] = std(samples[i], means[i])
        }
        return res
    }

    fun normalize(arr: List<FloatArray>, means: FloatArray, stds: FloatArray): List<FloatArray> {
        return arr.map { row ->
            row.mapIndexed { i, value -> (value - means[i]) / stds[i] }.toFloatArray()
        }.toList()
    }

    fun normalize(arr: FloatArray, mean: Float, std: Float): FloatArray {
        return arr.map { (it - mean) / std }.toFloatArray()
    }

    fun getNormalizationStats(xs: List<FloatArray>, ys: FloatArray, numFeatures: Int): NormalizationStats {
        val xMeans = mean(xs, numFeatures)
        val xStds = std(xs, xMeans)
        val yMean = mean(ys)
        val yStd = std(ys, yMean)
        return NormalizationStats(xMeans, xStds, yMean, yStd)
    }
}

data class NormalizationStats(
    val xMeans: FloatArray,
    val xStds: FloatArray,
    val yMean: Float,
    val yStd: Float
)

