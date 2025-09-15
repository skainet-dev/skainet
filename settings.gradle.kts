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

include("skainet-core:skainet-tensors-api")
include("skainet-core:skainet-tensors")
include("skainet-core:skainet-performance")
