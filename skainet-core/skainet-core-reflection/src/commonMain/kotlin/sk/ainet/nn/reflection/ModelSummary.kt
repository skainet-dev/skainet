package sk.ainet.nn.reflection

import sk.ainet.core.tensor.DType
import sk.ainet.core.tensor.Shape
import sk.ainet.core.tensor.Tensor
import sk.ainet.nn.Module
import sk.ainet.nn.reflection.table.table
import sk.ainet.nn.topology.ModuleParameters
import sk.ainet.nn.topology.by


public data class NodeSummary(val name: String, val input: Shape, val output: Shape, val params: Long)

public class Summary<T : DType, V> {

    public val nodes: MutableList<NodeSummary> = mutableListOf<NodeSummary>()

    private fun countParameters(module: Module<T, V>): Long {
        var params = 0L

        if (module is ModuleParameters<*, *>) {
            module.params.forEach { param ->
                params += param.value.shape.volume
            }
        }

        return params
    }

    private fun nodeSummary(module: Module<T, V>, input: Shape, output: Shape): NodeSummary {
        val params = countParameters(module)

        return NodeSummary(
            module.name,
            input,
            output,
            params
        )
    }


    private fun traverseModules(module: Module<T, V>, currentInput: Shape): List<NodeSummary> {
        val result = mutableListOf<NodeSummary>()
        
        // For leaf modules (modules with parameters), create a summary
        if (module is ModuleParameters<*, *> && module.params.isNotEmpty()) {
            // We'll use the current input shape as both input and output for now
            // In a real implementation, this would require actual forward pass
            val summary = nodeSummary(module, currentInput, currentInput)
            result.add(summary)
        }
        
        // Recursively traverse nested modules
        module.modules.forEach { subModule ->
            result.addAll(traverseModules(subModule, currentInput))
        }
        
        return result
    }

    public fun summary(model: Module<T, V>, input: Shape, batch_size: Int = -1): List<NodeSummary> {
        nodes.clear()
        val summaries = traverseModules(model, input)
        nodes.addAll(summaries)
        return summaries
    }

    public fun printSummary(nodes: List<NodeSummary>): String =
        table {
            cellStyle {
                border = true
            }
            header {
                row {
                    cell("Layer (type)")
                    cell("Output Shape")
                    cell("Param #")
                }
            }
            nodes.forEach { node ->
                row {
                    cell(node.name)
                    cell(node.output.toString())
                    cell(node.params)
                }
            }
        }.toString()
}

public fun <T : DType, V> Module<T, V>.describe(input: Shape, batch_size: Int = -1): String {
    val summary = Summary<T, V>()
    val nodes = summary.summary(this, input, batch_size)
    return summary.printSummary(nodes)
}
