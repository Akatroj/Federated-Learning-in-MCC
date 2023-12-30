package com.agh.federatedlearninginmcc.ml

typealias ModelVariantKey = Int

enum class ModelVariant(val key: ModelVariantKey, val modelConfig: ModelConfig, val trainerConfig: TrainerConfig) {
    LOCAL_TIME(
        0,
        ModelConfig(
            "local_time.tflite",
            "local_time.trained.tflite",
            "local_time",
            4,
            intArrayOf(4 * 5, 5, 5 * 1, 1), // weights1, biases1, weights2...
            ),
        TrainerConfig("localTime", 8885)
    ),
    CLOUD_COMPUTATION_TIME(
        1,
        ModelConfig(
            "cloud_computation_time.tflite",
            "cloud_computation_time.trained.tflite",
            "cloud_computation_time",
            5,
            intArrayOf(5 * 10, 10, 10 * 5, 5, 5 * 1, 1),
        ),
        TrainerConfig("cloudComputationTime", 8886)
    ),
    CLOUD_TRANSMISSION_TIME(
        2,
        ModelConfig(
            "cloud_transmission_time.tflite",
            "cloud_transmission_time.trained.tflite",
            "cloud_transmission_time",
            5,
            intArrayOf(5 * 10, 10, 10 * 5, 5, 5 * 1, 1),
        ),
        TrainerConfig("cloudTransmissionTime", 8887)
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
