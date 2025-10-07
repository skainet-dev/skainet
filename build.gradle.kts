plugins {
    alias(libs.plugins.androidLibrary) apply false
    alias(libs.plugins.kotlinMultiplatform) apply  false
    alias(libs.plugins.jetbrainsKotlinJvm) apply false
    alias(libs.plugins.vanniktech.mavenPublish) apply false
    alias(libs.plugins.kover)
}

allprojects {
    group = "sk.ainet"
}

kover {
    reports {
        total {
            html {
                onCheck = true
            }
            xml {
                onCheck = true
            }
        }
    }
}