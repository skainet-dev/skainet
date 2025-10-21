package sk.ainet.lang.graph

import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.DType

/**
 * Result of graph execution containing the computed result and execution metadata
 */
public data class GraphExecutionResult<T: DType, V, R>(
    val result: R,
    val graph: ComputeGraph,
    val tape: ExecutionTape?,
    val executionContext: ExecutionContext
) {
    /**
     * Execute the recorded graph with the current execution context
     */
    public fun execute(): R {
        // TODO: Implement graph execution
        return result
    }
    
    /**
     * Get execution statistics
     */
    public fun getStats(): ExecutionStats = executionContext.getExecutionStats()
}

/**
 * Global storage for current graph execution context
 * Note: In multiplatform common code, we use a simple global variable instead of ThreadLocal
 */
private var currentContext: GraphExecutionContext<*, *>? = null

@Suppress("UNCHECKED_CAST")
private fun <T: DType, V> currentGraphContext(): GraphExecutionContext<T, V> {
    return currentContext as? GraphExecutionContext<T, V>
        ?: error("No graph execution context available. Operations must be called within exec { } block")
}

private fun <T: DType, V> setCurrentContext(context: GraphExecutionContext<T, V>) {
    @Suppress("UNCHECKED_CAST")
    currentContext = context as GraphExecutionContext<*, *>
}

private fun clearCurrentContext() {
    currentContext = null
}

/**
 * Creates a graph execution context and executes the given block within it.
 * Similar to the network context pattern but for graph execution.
 * 
 * Usage:
 * val result = exec<FP32, Float> {
 *     a + b
 * }
 */
public fun <T: DType, V, R> exec(
    dataFactory: TensorDataFactory = DenseTensorDataFactory(),
    block: GraphExecutionContext<T, V>.() -> R
): GraphExecutionResult<T, V, R> {
    val context = DefaultGraphExecutionContext<T, V>(dataFactory)
    
    return try {
        setCurrentContext(context)
        context.executionContext.startRecording(context.executionContext.createTape())
        
        val result = context.block()
        val tape = context.executionContext.stopRecording()
        
        GraphExecutionResult(
            result = result,
            graph = context.graph,
            tape = tape,
            executionContext = context.executionContext
        )
    } finally {
        clearCurrentContext()
    }
}

/**
 * Internal function to access current graph context from operator extensions
 */
internal fun <T: DType, V> getCurrentGraphContext(): GraphExecutionContext<T, V> {
    return currentGraphContext<T, V>()
}