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

rootProject.name = "SKaiNET"

include("skainet-core:skainet-tensors-api")
include("skainet-core:skainet-tensors")
include("skainet-core:skainet-performance")
include("skainet-core:skainet-core-reflection")
include("skainet-nn:skainet-nn-api")
include("skainet-nn:skainet-nn-relection")
include("skainet-data:skainet-data-api")
include("skainet-data:skainet-data-simple")
