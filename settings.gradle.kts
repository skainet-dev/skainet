pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "skainet"
include(":core")
include(":io")
include(":gguf")
include(":model-zoo")
include(":samples:mnist-mlp-cli")
