plugins {
    alias(libs.plugins.androidLibrary) apply false
    alias(libs.plugins.kotlinMultiplatform) apply  false
    alias(libs.plugins.jetbrainsKotlinJvm) apply false
    alias(libs.plugins.vanniktech.mavenPublish) apply false
    alias(libs.plugins.kover)
    alias(libs.plugins.binary.compatibility.validator) apply false
    alias(libs.plugins.ksp) apply false
    alias(libs.plugins.asciidoctorJvm) apply false
    alias(libs.plugins.dokka) apply false
    id("sk.ainet.documentation")
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

// Custom task to generate operator documentation
tasks.register("generateOperatorDocs") {
    group = "documentation"
    description = "Generate operator documentation from KSP-generated JSON files"
    
    // Configure inputs for incremental builds
    inputs.files("skainet-lang/skainet-lang-core/build/generated/ksp/metadata/commonMain/resources/operators.json")
    inputs.files("tools/docgen/src/main/kotlin")
    
    // Configure outputs for incremental builds
    outputs.dir("docs/modules/operators/_generated_")
    outputs.cacheIf { true }
    
    // Depend on KSP processing
    dependsOn(":skainet-lang:skainet-lang-core:kspCommonMainKotlinMetadata")
    
    // Depend on DocGen application
    dependsOn(":tools:docgen:run")
    
    // Final step: process with AsciiDoctor
    finalizedBy(":tools:docgen:asciidoctor")
    
    doLast {
        println("Operator documentation generation completed")
    }
}

// Documentation plugin configuration
documentation {
    inputFile.set(file("skainet-lang/skainet-lang-core/build/generated/ksp/metadata/commonMain/resources/operators.json"))
    outputDirectory.set(file("docs/modules/operators/_generated_"))
    includeBackendStatus.set(true)
    generateIndex.set(true)
}

tasks.named("generateDocs") {
    dependsOn(":skainet-lang:skainet-lang-core:kspCommonMainKotlinMetadata")
}