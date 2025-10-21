package sk.ainet.lang.graph

import sk.ainet.lang.tensor.ops.GraphTensorOps
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.types.DType

/**
 * Context for graph execution DSL to define operations within a graph scope.
 * Similar to NetworkContext but focused on graph computation execution.
 */
public interface GraphExecutionContext<T: DType, V> {
    public val graphOps: GraphTensorOps<V>
    public val executionContext: ExecutionContext
    public val graph: ComputeGraph
}

/**
 * Default implementation of GraphExecutionContext
 */
public class DefaultGraphExecutionContext<T: DType, V>(
    private val dataFactory: TensorDataFactory
) : GraphExecutionContext<T, V> {
    
    override val executionContext: ExecutionContext = DefaultExecutionContext()
    override val graph: ComputeGraph = DefaultComputeGraph()
    
    private val baseOps = sk.ainet.lang.tensor.ops.VoidTensorOps<V>()
    override val graphOps: GraphTensorOps<V> = GraphTensorOps(baseOps, graph, executionContext)
    
    init {
        // Switch to graph mode and start recording by default
        executionContext.switchToGraph()
    }
}