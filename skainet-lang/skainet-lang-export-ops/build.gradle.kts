plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.kotlinSerialization)
    alias(libs.plugins.ksp)
}


group = "org.mikrograd.samples"

kotlin {

    compilerOptions {
        // Common compiler options applied to all Kotlin source sets
        freeCompilerArgs.add("-Xexpect-actual-classes")
        freeCompilerArgs.add("-Xmulti-platform")
    }

    jvmToolchain(17)

    jvm()


    sourceSets {
        commonMain.dependencies {
            implementation(project(":skainet-lang:skainet-lang-core"))
            implementation(libs.kotlinx.serialization.json)

        }

        commonTest.dependencies {
            implementation(kotlin("test-common"))
            implementation(kotlin("test-annotations-common"))
        }

        val jvmMain by getting {
            kotlin.srcDir("build/generated/ksp/jvm/jvmMain/kotlin")
            dependencies {
                implementation(project(":skainet-lang:skainet-lang-ksp-annotations"))
                implementation("com.networknt:json-schema-validator:1.0.87")
                implementation("com.fasterxml.jackson.core:jackson-databind:2.15.2")
            }
        }



        jvmTest.dependencies {
            implementation(kotlin("test-junit"))
        }
    }
}

dependencies {
    //    add("kspCommonMainMetadata", project(":test-processor"))
    add("kspJvm", project(":skainet-lang:skainet-lang-ksp-processor"))
}

ksp {
    arg("ksp.verbose", "true")
}

// Add a run task for the JVM application
tasks.register<JavaExec>("runJvm") {
    group = "application"
    description = "Run the JVM application"
    classpath = files(kotlin.jvm().compilations["main"].output.allOutputs, configurations.getByName("jvmRuntimeClasspath"))
    mainClass.set("com.example.MainKt")
}

// Add a run task for the KspMain application
tasks.register<JavaExec>("runKspMain") {
    group = "application"
    description = "Run the KspMain application"
    classpath = files(kotlin.jvm().compilations["main"].output.allOutputs, configurations.getByName("jvmRuntimeClasspath"))
    mainClass.set("com.example.KspMainKt")
}

// Add schema validation task
tasks.register<JavaExec>("validateOperatorSchema") {
    group = "verification"
    description = "Validate generated operator.json files against the JSON schema"
    classpath = files(kotlin.jvm().compilations["main"].output.allOutputs, configurations.getByName("jvmRuntimeClasspath"))
    mainClass.set("org.mikrograd.diff.ksp.SchemaValidationMainKt")
    
    // Set build directory as argument
    args(project.buildDir.absolutePath)
    
    // Depend on KSP compilation to ensure JSON files are generated first
    dependsOn("kspKotlinJvm")
    
    doFirst {
        println("Validating operator documentation JSON schema...")
    }
}
