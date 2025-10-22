plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.ksp)
    alias(libs.plugins.kotlinSerialization)
    alias(libs.plugins.vanniktech.mavenPublish)
}


kotlin {
    jvm()

    sourceSets {
        val commonMain by getting {
            dependencies {

                implementation(libs.kotlinx.serialization.json)
            }
        }
    }
}

