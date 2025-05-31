import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    id("com.android.library")
    //alias(libs.plugins.vanniktech.mavenPublish)
}

kotlin {
    jvm()
    androidTarget {
        compilations.all {
            kotlinOptions {
                jvmTarget = "1.8"
            }
        }
    }

    wasmJs {
        browser()
    }

    iosX64()
    iosArm64()
    iosSimulatorArm64()

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation(project(":core"))
                implementation(libs.kotlinx.io.core)
                implementation(libs.kotlinx.serialization.json)
                implementation("io.ktor:ktor-client-core:3.1.3")
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.8.1")
            }
        }

        val commonTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
            }
        }

        val jvmMain by getting {
            dependencies {
                implementation("io.ktor:ktor-client-cio:3.1.3")
                implementation("io.ktor:ktor-client-plugins:3.1.1")
                implementation("io.ktor:ktor-client-logging:3.1.3")
                implementation("io.ktor:ktor-client-content-negotiation:3.1.3")
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core-jvm:1.8.1")
                implementation("ch.qos.logback:logback-classic:1.4.14") // For logging
            }
        }

        val jvmTest by getting {
            dependencies {
                implementation(libs.kotlinx.coroutines)
            }
        }

        val androidMain by getting {
            dependencies {
                implementation("io.ktor:ktor-client-android:3.1.3")
            }
        }

        val wasmJsMain by getting {
            dependencies {
                implementation("io.ktor:ktor-client-js:3.1.3")
            }
        }

        val iosX64Main by getting
        val iosArm64Main by getting
        val iosSimulatorArm64Main by getting
        val iosMain by creating {
            dependsOn(commonMain)
            iosX64Main.dependsOn(this)
            iosArm64Main.dependsOn(this)
            iosSimulatorArm64Main.dependsOn(this)
            dependencies {
                implementation("io.ktor:ktor-client-darwin:3.1.3")
            }
        }
    }
}

android {
    namespace = "sk.ai.net.model.zoo"
    compileSdk = 34
    defaultConfig {
        minSdk = 24
    }
}
