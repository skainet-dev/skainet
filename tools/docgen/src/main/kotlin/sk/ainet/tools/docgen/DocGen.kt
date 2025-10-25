package sk.ainet.tools.docgen

import kotlinx.cli.*
import kotlinx.serialization.json.Json
import java.io.File
import java.time.Instant
import java.time.format.DateTimeFormatter

/**
 * Main documentation generator that converts JSON operator documentation to AsciiDoc format.
 * 
 * Usage: DocGen -i input.json -o output_directory
 */
object DocGen {
    
    private val json = Json { 
        ignoreUnknownKeys = true 
        prettyPrint = true
    }
    
    fun generateDocumentation(inputFile: File, outputDir: File) {
        println("Reading JSON from: ${inputFile.absolutePath}")
        
        val jsonContent = inputFile.readText()
        val module = json.decodeFromString<OperatorDocModule>(jsonContent)
        
        println("Parsed module: ${module.module} with ${module.operators.size} operators")
        
        // Create output directory structure
        outputDir.mkdirs()
        val generatedDir = File(outputDir, "_generated_")
        generatedDir.mkdirs()
        
        // Generate main index page
        generateMainIndex(module, generatedDir)
        
        // Generate individual operator pages
        module.operators.forEach { operator ->
            generateOperatorPage(operator, module, generatedDir)
        }
        
        println("Generated documentation in: ${generatedDir.absolutePath}")
    }
    
    private fun generateMainIndex(module: OperatorDocModule, outputDir: File) {
        val content = buildString {
            appendLine("= ${module.module} Operators")
            appendLine()
            appendLine("// Generated on ${formatTimestamp(module.timestamp)}")
            appendLine("// Version: ${module.version}")
            appendLine("// Commit: ${module.commit}")
            appendLine()
            appendLine("This documentation is automatically generated from the codebase annotations.")
            appendLine()
            appendLine("== Operators")
            appendLine()
            
            // Group operators by modality
            val operatorsByModality = module.operators.groupBy { it.modality }
            operatorsByModality.entries.sortedBy { it.key }.forEach { (modality, operators) ->
                appendLine("=== ${modality.capitalize()} Operators")
                appendLine()
                operators.sortedBy { it.name }.forEach { operator ->
                    appendLine("* xref:${operator.name.lowercase()}.adoc[${operator.name}] - ${operator.packageName}")
                }
                appendLine()
            }
        }
        
        File(outputDir, "index.adoc").writeText(content)
    }
    
    private fun generateOperatorPage(operator: OperatorDoc, module: OperatorDocModule, outputDir: File) {
        val content = buildString {
            appendLine("= ${operator.name}")
            appendLine()
            appendLine("// Generated on ${formatTimestamp(module.timestamp)}")
            appendLine("// Package: ${operator.packageName}")
            appendLine("// Modality: ${operator.modality}")
            appendLine()
            appendLine("Package: `${operator.packageName}`")
            appendLine()
            appendLine("Modality: *${operator.modality}*")
            appendLine()
            
            if (operator.functions.isNotEmpty()) {
                appendLine("== Functions")
                appendLine()
                
                operator.functions.sortedBy { it.name }.forEach { function ->
                    generateFunctionSection(function, this)
                }
            }
        }
        
        File(outputDir, "${operator.name.lowercase()}.adoc").writeText(content)
    }
    
    private fun generateFunctionSection(function: FunctionDoc, builder: StringBuilder) {
        builder.apply {
            appendLine("=== ${function.name}")
            appendLine()
            appendLine("[source,kotlin]")
            appendLine("----")
            appendLine(function.signature)
            appendLine("----")
            appendLine()
            
            // Parameters table
            if (function.parameters.isNotEmpty()) {
                appendLine("==== Parameters")
                appendLine()
                appendLine("[cols=\"1,2,3\"]")
                appendLine("|===")
                appendLine("| Name | Type | Description")
                appendLine()
                function.parameters.forEach { param ->
                    appendLine("| ${param.name}")
                    appendLine("| `${param.type}`")
                    appendLine("| ${param.description.ifEmpty { "_No description_" }}")
                    appendLine()
                }
                appendLine("|===")
                appendLine()
            }
            
            // Return type
            appendLine("==== Returns")
            appendLine()
            appendLine("`${function.returnType}`")
            appendLine()
            
            // Backend status table
            if (function.statusByBackend.isNotEmpty()) {
                appendLine("==== Backend Status")
                appendLine()
                generateBackendStatusTable(function, this)
            }
            
            // API reference link
            appendLine("==== See also")
            appendLine()
            appendLine("* xref:api:${function.name}[API Reference (Dokka)]")
            appendLine("* xref:theory:${function.name}.adoc[Mathematical Definition]")
            appendLine("* xref:examples:${function.name}.adoc[Usage Examples]")
            appendLine()
        }
    }
    
    private fun generateBackendStatusTable(function: FunctionDoc, builder: StringBuilder) {
        builder.apply {
            appendLine("[cols=\"1,1,2\"]")
            appendLine("|===")
            appendLine("| Backend | Status | Notes")
            appendLine()
            
            function.statusByBackend.entries.sortedBy { it.key }.forEach { (backend, status) ->
                appendLine("| ${backend}")
                appendLine("| ${formatStatus(status)}")
                
                val backendNotes = function.notes.filter { it.backend == backend }
                if (backendNotes.isNotEmpty()) {
                    val notesText = backendNotes.joinToString(", ") { note ->
                        when (note.type) {
                            "owner" -> "Owner: ${note.message}"
                            "issue" -> "Issue: ${note.message}"
                            else -> "${note.type}: ${note.message}"
                        }
                    }
                    appendLine("| ${notesText}")
                } else {
                    appendLine("| _None_")
                }
                appendLine()
            }
            appendLine("|===")
            appendLine()
        }
    }
    
    private fun formatStatus(status: String): String {
        return when (status) {
            "implemented" -> "✅ Implemented"
            "not_implemented" -> "❌ Not Implemented"
            "in_progress" -> "🚧 In Progress"
            else -> status
        }
    }
    
    private fun formatTimestamp(timestamp: String): String {
        return try {
            val instant = Instant.parse(timestamp)
            DateTimeFormatter.ISO_LOCAL_DATE_TIME.format(instant.atZone(java.time.ZoneId.systemDefault()))
        } catch (e: Exception) {
            timestamp
        }
    }
    
    private fun String.capitalize(): String {
        return this.replaceFirstChar { if (it.isLowerCase()) it.titlecase() else it.toString() }
    }
}

fun main(args: Array<String>) {
    val parser = ArgParser("docgen")
    val input by parser.option(ArgType.String, shortName = "i", description = "Input JSON file").required()
    val output by parser.option(ArgType.String, shortName = "o", description = "Output directory").required()
    
    parser.parse(args)
    
    val inputFile = File(input)
    val outputDir = File(output)
    
    if (!inputFile.exists()) {
        println("Error: Input file does not exist: $input")
        return
    }
    
    DocGen.generateDocumentation(inputFile, outputDir)
}