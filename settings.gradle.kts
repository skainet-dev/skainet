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

include("skainet-lang:skainet-lang-core")
include("skainet-lang:skainet-lang-models")
include("skainet-lang:skainet-lang-ksp-annotations")
include("skainet-lang:skainet-lang-ksp-processor")
include("skainet-lang:skainet-lang-export-ops")
