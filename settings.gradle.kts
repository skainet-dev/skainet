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

include("skainet-lang:skainet-lang-api")
include("skainet-lang:skainet-lang-memory")
