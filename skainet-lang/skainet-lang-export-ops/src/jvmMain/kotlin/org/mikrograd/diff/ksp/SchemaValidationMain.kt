package org.mikrograd.diff.ksp

import java.io.File
import kotlin.system.exitProcess

/**
 * Main entry point for schema validation task.
 * 
 * This is executed by the Gradle validateOperatorSchema task to validate
 * generated operator.json files against the JSON schema.
 */
fun main(args: Array<String>) {
    if (args.isEmpty()) {
        println("Error: Build directory path required as argument")
        exitProcess(1)
    }
    
    val buildDirPath = args[0]
    val buildDir = File(buildDirPath)
    
    println("Starting schema validation for operator documentation...")
    println("Build directory: ${buildDir.absolutePath}")
    
    val validationResults = SchemaValidator.validateBuildOutput(buildDir)
    
    if (validationResults.isEmpty()) {
        println("Warning: No validation results returned")
        exitProcess(1)
    }
    
    var hasErrors = false
    var totalFiles = 0
    var validFiles = 0
    
    for (result in validationResults) {
        totalFiles++
        
        if (result.result.isValid) {
            validFiles++
            println("✓ VALID: ${result.file.relativeTo(buildDir)}")
        } else {
            hasErrors = true
            println("✗ INVALID: ${result.file.relativeTo(buildDir)}")
            println("  Errors:")
            for (error in result.result.errors) {
                println("    - $error")
            }
        }
    }
    
    println("\n" + "=".repeat(60))
    println("Schema Validation Summary")
    println("=".repeat(60))
    println("Total files validated: $totalFiles")
    println("Valid files: $validFiles")
    println("Invalid files: ${totalFiles - validFiles}")
    
    if (hasErrors) {
        println("\n❌ Schema validation FAILED")
        println("Please fix the validation errors above and run again.")
        exitProcess(1)
    } else {
        println("\n✅ All operator documentation files are valid!")
        println("Schema validation PASSED")
        exitProcess(0)
    }
}