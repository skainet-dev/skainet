package sk.ainet.lang.tools.export.ops

import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.networknt.schema.JsonSchema
import com.networknt.schema.JsonSchemaFactory
import com.networknt.schema.SpecVersion
import java.io.File
import java.io.InputStream

/**
 * Utility class for validating operator documentation JSON against the JSON schema.
 */
object SchemaValidator {
    
    private val objectMapper = ObjectMapper()
    private val schemaFactory = JsonSchemaFactory.getInstance(SpecVersion.VersionFlag.V202012)
    
    /**
     * Validates a JSON file against the operator documentation schema.
     * 
     * @param jsonFile The JSON file to validate
     * @return ValidationResult containing success status and any errors
     */
    fun validateFile(jsonFile: File): ValidationResult {
        return try {
            if (!jsonFile.exists()) {
                return ValidationResult(false, listOf("File does not exist: ${jsonFile.absolutePath}"))
            }
            
            val jsonNode = objectMapper.readTree(jsonFile)
            validate(jsonNode)
        } catch (e: Exception) {
            ValidationResult(false, listOf("Error reading JSON file: ${e.message}"))
        }
    }
    
    /**
     * Validates a JSON string against the operator documentation schema.
     * 
     * @param jsonContent The JSON content as a string
     * @return ValidationResult containing success status and any errors
     */
    fun validateContent(jsonContent: String): ValidationResult {
        return try {
            val jsonNode = objectMapper.readTree(jsonContent)
            validate(jsonNode)
        } catch (e: Exception) {
            ValidationResult(false, listOf("Error parsing JSON content: ${e.message}"))
        }
    }
    
    /**
     * Validates a JsonNode against the operator documentation schema.
     * 
     * @param jsonNode The JsonNode to validate
     * @return ValidationResult containing success status and any errors
     */
    private fun validate(jsonNode: JsonNode): ValidationResult {
        return try {
            val schema = loadSchema()
            val errors = schema.validate(jsonNode)
            
            if (errors.isEmpty()) {
                ValidationResult(true, emptyList())
            } else {
                val errorMessages = errors.map { error ->
                    "${error.path}: ${error.message}"
                }
                ValidationResult(false, errorMessages)
            }
        } catch (e: Exception) {
            ValidationResult(false, listOf("Schema validation error: ${e.message}"))
        }
    }
    
    /**
     * Loads the JSON schema from resources.
     * 
     * @return JsonSchema instance
     */
    private fun loadSchema(): JsonSchema {
        val schemaStream = getSchemaStream()
            ?: throw IllegalStateException("Cannot find schema resource: schemas/operator-doc-schema-v1.json")
        
        return schemaFactory.getSchema(schemaStream)
    }
    
    /**
     * Gets the schema file as an InputStream from resources.
     * 
     * @return InputStream for the schema file or null if not found
     */
    private fun getSchemaStream(): InputStream? {
        return this::class.java.classLoader.getResourceAsStream("schemas/operator-doc-schema-v1.json")
    }
    
    /**
     * Validates all operator.json files in the given directory recursively.
     * 
     * @param buildDir The build directory to search for operator.json files
     * @return List of ValidationResult for each file found
     */
    fun validateBuildOutput(buildDir: File): List<FileValidationResult> {
        val results = mutableListOf<FileValidationResult>()
        
        if (!buildDir.exists()) {
            return listOf(FileValidationResult(buildDir, ValidationResult(false, listOf("Build directory does not exist"))))
        }
        
        val operatorJsonFiles = buildDir.walkTopDown()
            .filter { it.isFile && it.name == "operators.json" }
            .toList()
        
        if (operatorJsonFiles.isEmpty()) {
            return listOf(FileValidationResult(buildDir, ValidationResult(false, listOf("No operators.json files found in build directory"))))
        }
        
        for (file in operatorJsonFiles) {
            val result = validateFile(file)
            results.add(FileValidationResult(file, result))
        }
        
        return results
    }
}

/**
 * Result of JSON schema validation.
 * 
 * @param isValid Whether the validation passed
 * @param errors List of validation error messages
 */
data class ValidationResult(
    val isValid: Boolean,
    val errors: List<String>
) {
    /**
     * Returns a formatted string of all errors.
     */
    fun getErrorsAsString(): String {
        return errors.joinToString("\n")
    }
}

/**
 * Result of validating a specific file.
 * 
 * @param file The file that was validated
 * @param result The validation result
 */
data class FileValidationResult(
    val file: File,
    val result: ValidationResult
)