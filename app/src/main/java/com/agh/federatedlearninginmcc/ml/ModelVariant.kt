package com.agh.federatedlearninginmcc.ml

typealias ModelVariantKey = Int

enum class ModelVariant(val key: ModelVariantKey, val modelConfig: ModelConfig, val trainerConfig: TrainerConfig) {
    LOCAL_TIME(
        0,
        ModelConfig(
            "local_time.tflite",
            "local_time_v6.trained.tflite",
            "local_time",
            dontStandardizeDims = setOf(0),
            inputDimensions = 6,
            layerSizes = intArrayOf(6 * 16, 16, 16 * 8, 8, 8 * 4, 4, 4 * 1, 1),
            ),
        TrainerConfig("localTime", 8885)
    ),
    CLOUD_COMPUTATION_TIME(
        1,
        ModelConfig(
            "cloud_computation_time.tflite",
            "cloud_computation_time_v3.trained.tflite",
            "cloud_computation_time",
            dontStandardizeDims = setOf(0, 6, 7),
            inputDimensions = 8,
            layerSizes = intArrayOf(8 * 16, 16, 16 * 8, 8, 8 * 4, 4, 4 * 1, 1),
        ),
        TrainerConfig("cloudComputationTime", 8886)
    ),
    CLOUD_TRANSMISSION_TIME(
        2,
        ModelConfig(
            "cloud_transmission_time.tflite",
            "cloud_transmission_time_v3.trained.tflite",
            "cloud_transmission_time",
            dontStandardizeDims = setOf(0, 2, 3, 4),
            inputDimensions = 5,
            layerSizes = intArrayOf(5 * 16, 16, 16 * 8, 8, 8 * 1, 1),
        ),
        TrainerConfig("cloudTransmissionTime", 8887)
    )
}

data class ModelConfig(
    val modelAssetsFile: String,
    val modelTrainedFile: String,
    val name: String,
    val dontStandardizeDims: Set<Int>,
    // TODO try not to hardcode these two
    val inputDimensions: Int,
    val layerSizes: IntArray,
)

data class TrainerConfig(
    val tag: String,
    val port: Int,
)
