package sk.ainet.lang.graph

/**
 * Represents a node in the computational graph containing operation metadata.
 * Each node represents a single operation with its inputs, outputs, and parameters.
 */
public data class GraphNode(
    /**
     * Unique identifier for this node
     */
    public val id: String,
    
    /**
     * The operation this node represents
     */
    public val operation: Operation,
    
    /**
     * Metadata about inputs to this operation
     */
    public val inputs: List<TensorSpec>,
    
    /**
     * Metadata about outputs from this operation
     */
    public val outputs: List<TensorSpec>,
    
    /**
     * Additional metadata for this node
     */
    public val metadata: Map<String, Any> = emptyMap()
) {
    
    /**
     * Gets the operation name
     */
    public val operationName: String get() = operation.name
    
    /**
     * Gets the operation type
     */
    public val operationType: String get() = operation.type
    
    override fun toString(): String {
        return "GraphNode(id=$id, operation=${operation.name}, inputs=${inputs.size}, outputs=${outputs.size})"
    }
}

/**
 * Specification for a tensor in the graph, containing shape and type information
 */
public data class TensorSpec(
    /**
     * Name/identifier of this tensor
     */
    public val name: String,
    
    /**
     * Shape of the tensor (null for dynamic shapes)
     */
    public val shape: List<Int>?,
    
    /**
     * Data type of the tensor
     */
    public val dtype: String,
    
    /**
     * Whether this tensor requires gradients
     */
    public val requiresGrad: Boolean = false,
    
    /**
     * Additional metadata for this tensor
     */
    public val metadata: Map<String, Any> = emptyMap()
)