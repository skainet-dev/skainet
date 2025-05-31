plugins {
    kotlin("multiplatform")
}

// Disable default hierarchy template to avoid warnings
kotlin.sourceSets.all {
    languageSettings.optIn("kotlin.RequiresOptIn")
}

kotlin {
    // Configure JVM toolchain at the extension level
    jvmToolchain(11)

    jvm {
        // JVM target configuration
        compilations.all {
            compileTaskProvider.configure {
                compilerOptions.jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_11)
            }
        }
    }

    // Only include JVM target for now to simplify the build
    // We'll add native targets back once we resolve the dependency issues

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation(project(":io"))
                implementation(project(":gguf"))
                implementation(project(":core"))
                implementation(project(":model-zoo"))
                implementation(libs.kotlinx.io.core)
                implementation(libs.kotlinx.coroutines)

                implementation(libs.kotlinx.cli)

                implementation(libs.ktor.client.plugins)
                implementation(libs.ktor.client.logging)

            }
        }

        val jvmMain by getting {
            dependencies {
                implementation(libs.ktor.client.cio)
            }
        }
    }
}

// Configure JVM jar task
tasks.register<Jar>("jvmFatJar") {
    dependsOn(tasks.named("jvmJar"))
    archiveClassifier.set("fat")

    manifest {
        attributes(
            "Main-Class" to "sk.ai.net.samples.mnist.mlp.MainKt"
        )
    }

    from(
        configurations.named("jvmRuntimeClasspath").get().map { 
            if (it.isDirectory) it else zipTree(it) 
        }
    )

    with(tasks.named<Jar>("jvmJar").get())

    // Exclude META-INF files from the dependencies
    exclude("META-INF/*.SF", "META-INF/*.DSA", "META-INF/*.RSA")

    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
}
