package com.agh.federatedlearninginmcc.ml

typealias ModelVariantKey = Int

enum class ModelVariant(val key: ModelVariantKey, val modelConfig: ModelConfig, val trainerConfig: TrainerConfig) {
    LOCAL_TIME(
        0,
        ModelConfig(
            "local_time.tflite",
            "local_time.trained.tflite",
            "local_time",
            3,
            intArrayOf(3 * 5, 5, 5 * 1, 1) // weights1, biases1, weights2...
        ),
        TrainerConfig("localTime", 8885)
    ),
    CLOUD_TIME(
        1,
        ModelConfig(
            "cloud_time.tflite",
            "cloud_time.trained.tflite",
            "cloud_time",
            3,
            intArrayOf(3 * 10, 10, 10 * 5, 5, 5 * 1, 1)
        ),
        TrainerConfig("cloudTime", 8886)
    )
}

data class ModelConfig(
    val modelAssetsFile: String,
    val modelTrainedFile: String,
    val name: String,
    // TODO try not to hardcode these two
    val inputDimensions: Int,
    val layerSizes: IntArray,
)

data class TrainerConfig(
    val tag: String,
    val port: Int,
)
