plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.ksp)
    id("com.vanniktech.maven.publish")
}


kotlin {
    jvm()
}