import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.node.ArrayNode
import com.fasterxml.jackson.databind.node.ObjectNode
import com.networknt.schema.JsonSchemaFactory
import com.networknt.schema.SpecVersion
import org.gradle.api.DefaultTask
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.tasks.*

@CacheableTask
abstract class SchemaValidationTask : DefaultTask() {
    @get:InputDirectory
    @get:Optional
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val searchDirectory: DirectoryProperty

    init {
        // Default to the root project directory to cover all subprojects by default
        searchDirectory.convention(project.rootProject.layout.projectDirectory)
    }

    private fun normalizeForSchema(root: JsonNode): JsonNode {
        if (root is ObjectNode) {
            // Normalize operators array
            val operators = root.get("operators")
            if (operators is ArrayNode) {
                operators.forEach { opNode ->
                    if (opNode is ObjectNode) {
                        // Handle legacy "package" -> "packageName"
                        if (!opNode.has("packageName") && opNode.has("package")) {
                            opNode.set<JsonNode>("packageName", opNode.get("package"))
                            opNode.remove("package")
                        }
                        // Normalize functions -> notes
                        val functions = opNode.get("functions")
                        if (functions is ArrayNode) {
                            functions.forEach { fnNode ->
                                if (fnNode is ObjectNode) {
                                    val notes = fnNode.get("notes")
                                    if (notes is ArrayNode) {
                                        notes.forEach { noteNode ->
                                            if (noteNode is ObjectNode) {
                                                if (!noteNode.has("content") && noteNode.has("message")) {
                                                    noteNode.set<JsonNode>("content", noteNode.get("message"))
                                                    noteNode.remove("message")
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return root
    }

    @TaskAction
    fun validate() {
        val buildDir = (if (searchDirectory.isPresent) searchDirectory.get() else project.rootProject.layout.projectDirectory).asFile
        val schemaStream = this::class.java.classLoader.getResourceAsStream("schemas/operator-doc-schema-v1.json")
            ?: throw IllegalStateException("Cannot find schema resource: schemas/operator-doc-schema-v1.json")
        val schema = JsonSchemaFactory.getInstance(SpecVersion.VersionFlag.V202012).getSchema(schemaStream)

        if (!buildDir.exists()) {
            throw RuntimeException("Build directory does not exist: ${buildDir.absolutePath}")
        }

        val operatorJsonFiles = buildDir.walkTopDown()
            .filter { it.isFile && it.name == "operators.json" }
            .toList()

        if (operatorJsonFiles.isEmpty()) {
            logger.lifecycle("No operators.json files found under: ${buildDir.absolutePath}. Skipping schema validation.")
            return
        }

        var total = 0
        var valid = 0
        val errors = mutableListOf<String>()

        // Create ObjectMapper locally to keep task configuration-cache friendly
        val objectMapper = ObjectMapper()

        operatorJsonFiles.forEach { file ->
            total++
            val original: JsonNode = objectMapper.readTree(file)
            val node = normalizeForSchema(original)
            val validationErrors = schema.validate(node)
            if (validationErrors.isEmpty()) {
                valid++
                logger.lifecycle("✓ VALID: ${file.relativeTo(buildDir)}")
            } else {
                logger.error("✗ INVALID: ${file.relativeTo(buildDir)}")
                validationErrors.forEach { err ->
                    val msg = "  - ${err.path}: ${err.message}"
                    errors.add("${file.relativeTo(buildDir)}: ${err.message}")
                    logger.error(msg)
                }
            }
        }

        logger.lifecycle("============================================")
        logger.lifecycle("Schema Validation Summary")
        logger.lifecycle("Total files: $total  Valid: $valid  Invalid: ${total - valid}")

        if (errors.isNotEmpty()) {
            throw RuntimeException("Schema validation failed for ${errors.size} issue(s). See log for details.")
        }
    }
}
