plugins {
    `kotlin-dsl`
    kotlin("jvm") version "2.2.20"
    kotlin("plugin.serialization") version "2.2.20"
}

repositories {
    gradlePluginPortal()
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.9.0")
    implementation("org.asciidoctor:asciidoctorj:3.0.0")
    implementation(gradleApi())
}

kotlin {
    jvmToolchain(17)
}