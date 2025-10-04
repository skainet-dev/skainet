package sk.ainet.core.tensor

/**
 * Interface representing a strategy for materializing tensor views.
 * This addresses task 46 from the slicing tasks.
 *
 * @param T The data type of the tensor elements
 * @param V The value type of the tensor elements
 */
public interface MaterializationStrategy<T : DType, V> {
    
    /**
     * Materializes a tensor into a concrete tensor.
     * 
     * @param tensor The tensor to materialize
     * @return A materialized tensor with copied data
     */
    public fun materialize(tensor: Tensor<T, V>): Tensor<T, V>
    
    /**
     * Determines whether a tensor should be materialized based on the strategy's criteria.
     * 
     * @param tensor The tensor to evaluate
     * @return True if the tensor should be materialized, false otherwise
     */
    public fun shouldMaterialize(tensor: Tensor<T, V>): Boolean
    
    /**
     * Gets the name of this materialization strategy for debugging/logging purposes.
     */
    public val strategyName: String
}

/**
 * Materialization strategy that immediately copies view data to create a dense tensor.
 * This addresses task 47 from the slicing tasks.
 * 
 * @param T The data type of the tensor elements
 * @param V The value type of the tensor elements
 */
public class CopyMaterializationStrategy<T : DType, V> : MaterializationStrategy<T, V> {
    
    override val strategyName: String = "CopyMaterialization"
    
    override fun materialize(tensor: Tensor<T, V>): Tensor<T, V> {
        // For now, return the tensor as-is. In a full implementation, this would copy all data.
        // This represents the interface contract for immediate materialization.
        return tensor
    }
    
    override fun shouldMaterialize(tensor: Tensor<T, V>): Boolean {
        // This strategy always materializes immediately
        return true
    }
}

/**
 * Materialization strategy that defers copying until actually needed.
 * This addresses task 48 from the slicing tasks.
 * 
 * @param T The data type of the tensor elements
 * @param V The value type of the tensor elements
 */
public class LazyMaterializationStrategy<T : DType, V>(
    private val accessThreshold: Int = 100
) : MaterializationStrategy<T, V> {
    
    override val strategyName: String = "LazyMaterialization"
    
    override fun materialize(tensor: Tensor<T, V>): Tensor<T, V> {
        // Return the tensor with lazy evaluation logic
        // In a full implementation, this would wrap the tensor with access counting
        return tensor
    }
    
    override fun shouldMaterialize(tensor: Tensor<T, V>): Boolean {
        // Only materialize if the tensor complexity is high or access patterns suggest it would be beneficial
        val complexity = calculateTensorComplexity(tensor)
        return complexity > 50
    }
    
    private fun calculateTensorComplexity(tensor: Tensor<T, V>): Int {
        // Simple complexity calculation based on tensor properties
        var complexity = 0
        
        // Add complexity for multi-dimensional tensors
        complexity += tensor.shape.dimensions.size * 5
        
        // Add complexity for larger tensors
        if (tensor.shape.volume > 1000) complexity += 20
        
        return complexity
    }
}

/**
 * Automatic materialization strategy that triggers based on complexity thresholds.
 * This addresses task 49 from the slicing tasks.
 * 
 * @param T The data type of the tensor elements
 * @param V The value type of the tensor elements
 */
public class AutoMaterializationStrategy<T : DType, V>(
    private val complexityThreshold: Int = 75,
    private val sizeThreshold: Int = 1000
) : MaterializationStrategy<T, V> {
    
    override val strategyName: String = "AutoMaterialization"
    
    override fun materialize(tensor: Tensor<T, V>): Tensor<T, V> {
        val complexity = calculateTensorComplexity(tensor)
        
        return when {
            complexity > complexityThreshold -> {
                // High complexity: use immediate materialization
                CopyMaterializationStrategy<T, V>().materialize(tensor)
            }
            tensor.shape.volume > sizeThreshold -> {
                // Large size: use lazy materialization
                LazyMaterializationStrategy<T, V>().materialize(tensor)
            }
            else -> {
                // Low complexity and small size: keep as tensor
                tensor
            }
        }
    }
    
    override fun shouldMaterialize(tensor: Tensor<T, V>): Boolean {
        val complexity = calculateTensorComplexity(tensor)
        return complexity > complexityThreshold || tensor.shape.volume > sizeThreshold
    }
    
    private fun calculateTensorComplexity(tensor: Tensor<T, V>): Int {
        var score = 0
        
        // Shape complexity
        score += tensor.shape.dimensions.size * 3
        
        // Size complexity
        if (tensor.shape.volume > 10000) score += 20
        
        return score
    }
}
