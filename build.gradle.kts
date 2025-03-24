plugins {
    alias(libs.plugins.androidLibrary) apply false
    alias(libs.plugins.kotlinMultiplatform) apply  false
    alias(libs.plugins.jetbrainsKotlinJvm) apply false
    alias(libs.plugins.binaryCompatibility) apply false
    alias(libs.plugins.modulegraph.souza) apply true
    alias(libs.plugins.spotless) apply false
}

allprojects {
    group = "sk.ai.net"
    version = "0.0.5"
}

moduleGraphConfig {
    readmePath.set("./Modules.md")
    heading = "### Module Graph"
}
