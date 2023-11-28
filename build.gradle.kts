plugins {
    id("com.android.application") version "8.1.0" apply false
    id("org.jetbrains.kotlin.android") version "1.9.0" apply false
    kotlin("jvm") version "1.9.10" apply false
    id("com.android.library") version "8.1.0" apply false
    id("com.google.protobuf") version "0.9.4" apply false
}

ext["grpcVersion"] = "1.57.2"
ext["grpcKotlinVersion"] = "1.4.0" // CURRENT_GRPC_KOTLIN_VERSION
ext["protobufVersion"] = "3.24.1"
ext["coroutinesVersion"] = "1.7.3"

subprojects {
    repositories {
        mavenLocal() // For testing new releases of gRPC Kotlin
        mavenCentral()
        google()
    }
}
