plugins {
    alias(libs.plugins.androidLibrary) apply false
    alias(libs.plugins.kotlinMultiplatform) apply  false
    alias(libs.plugins.jetbrainsKotlinJvm) apply false
    alias(libs.plugins.binaryCompatibility) apply false
    alias(libs.plugins.modulegraph.souza) apply true


}

allprojects {
    version = "0.0.3"
}

moduleGraphConfig {
    readmePath.set("./Modules.md")
    heading = "### Module Graph"
}
