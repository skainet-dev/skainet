plugins {
    alias(libs.plugins.androidLibrary) apply false
    alias(libs.plugins.kotlinMultiplatform) apply  false
    alias(libs.plugins.jetbrainsKotlinJvm) apply false
    alias(libs.plugins.vanniktech.mavenPublish) apply false
}

allprojects {
    group = "sk.ai.net"
    version = "0.0.7"
}