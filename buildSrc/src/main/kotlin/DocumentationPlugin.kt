import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.Action

class DocumentationPlugin : Plugin<Project> {
    override fun apply(project: Project) {
        val extension = project.extensions.create("documentation", DocumentationExtension::class.java, project)
        
        project.tasks.register("generateDocs", GenerateDocumentationTask::class.java, object : Action<GenerateDocumentationTask> {
            override fun execute(task: GenerateDocumentationTask) {
                task.group = "documentation"
                task.description = "Generate documentation from KSP metadata"
                
                task.inputFile.set(extension.inputFile)
                task.outputDirectory.set(extension.outputDirectory)
                task.templateDirectory.set(extension.templateDirectory)
                task.format.set(extension.format)
                task.includeBackendStatus.set(extension.includeBackendStatus)
                task.generateIndex.set(extension.generateIndex)
            }
        })
    }
}