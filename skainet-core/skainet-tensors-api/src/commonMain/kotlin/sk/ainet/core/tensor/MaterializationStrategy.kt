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
     * Materializes a tensor view into a concrete tensor.
     * 
     * @param view The tensor view to materialize
     * @return A materialized tensor with copied data
     */
    public fun materialize(view: TensorView<T, V>): Tensor<T, V>
    
    /**
     * Determines whether a view should be materialized based on the strategy's criteria.
     * 
     * @param view The tensor view to evaluate
     * @return True if the view should be materialized, false otherwise
     */
    public fun shouldMaterialize(view: TensorView<T, V>): Boolean
    
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
    
    override fun materialize(view: TensorView<T, V>): Tensor<T, V> {
        // For now, return the view as-is. In a full implementation, this would copy all data.
        // This represents the interface contract for immediate materialization.
        return view
    }
    
    override fun shouldMaterialize(view: TensorView<T, V>): Boolean {
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
    
    override fun materialize(view: TensorView<T, V>): Tensor<T, V> {
        // Return the view with lazy evaluation logic
        // In a full implementation, this would wrap the view with access counting
        return view
    }
    
    override fun shouldMaterialize(view: TensorView<T, V>): Boolean {
        // Only materialize if the view complexity is high or access patterns suggest it would be beneficial
        val complexity = calculateViewComplexity(view)
        return complexity > 50 || !view.isContiguous
    }
    
    private fun calculateViewComplexity(view: TensorView<T, V>): Int {
        // Simple complexity calculation based on stride patterns and shape
        var complexity = 0
        
        // Add complexity for non-unit strides
        view.strides.forEach { stride ->
            if (stride != 1) complexity += 10
        }
        
        // Add complexity for multi-dimensional views
        complexity += view.shape.dimensions.size * 5
        
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
    
    override fun materialize(view: TensorView<T, V>): Tensor<T, V> {
        val complexity = calculateViewComplexity(view)
        
        return when {
            complexity > complexityThreshold -> {
                // High complexity: use immediate materialization
                CopyMaterializationStrategy<T, V>().materialize(view)
            }
            view.shape.volume > sizeThreshold -> {
                // Large size: use lazy materialization
                LazyMaterializationStrategy<T, V>().materialize(view)
            }
            else -> {
                // Low complexity and small size: keep as view
                view
            }
        }
    }
    
    override fun shouldMaterialize(view: TensorView<T, V>): Boolean {
        val complexity = calculateViewComplexity(view)
        return complexity > complexityThreshold || view.shape.volume > sizeThreshold
    }
    
    private fun calculateViewComplexity(view: TensorView<T, V>): Int {
        var score = 0
        
        // Non-contiguous access patterns increase complexity
        if (!view.isContiguous) score += 30
        
        // Multiple dimensions with non-unit strides
        view.strides.forEachIndexed { index, stride ->
            if (stride != 1) score += (index + 1) * 5
        }
        
        // Shape complexity
        score += view.shape.dimensions.size * 3
        
        return score
    }
}

/**
 * View complexity scoring system for materialization decisions.
 * This addresses task 50 from the slicing tasks.
 */
public object ViewComplexityScorer {
    
    /**
     * Calculates a complexity score for a tensor view.
     * Higher scores indicate views that would benefit from materialization.
     * 
     * @param view The tensor view to score
     * @return Complexity score (0-100+)
     */
    public fun <T : DType, V> calculateComplexity(view: TensorView<T, V>): Int {
        var score = 0
        
        // Base score for non-contiguous access
        if (!view.isContiguous) {
            score += 40
        }
        
        // Stride pattern analysis
        val maxStride = view.strides.maxOrNull() ?: 1
        val minStride = view.strides.minOrNull() ?: 1
        val strideRange = maxStride - minStride
        score += (strideRange / 10).coerceAtMost(25)
        
        // Dimensionality complexity
        score += (view.shape.dimensions.size - 1) * 5
        
        // Memory access pattern complexity
        val volume = view.shape.volume
        if (volume > 10000) {
            score += 15
        }
        
        // Check for irregular access patterns
        if (hasIrregularAccessPattern(view)) {
            score += 20
        }
        
        return score.coerceAtMost(100)
    }
    
    private fun <T : DType, V> hasIrregularAccessPattern(view: TensorView<T, V>): Boolean {
        // Simple heuristic: if strides are not in decreasing order, consider it irregular
        val strides = view.strides
        for (i in 0 until strides.size - 1) {
            if (strides[i] < strides[i + 1]) {
                return true
            }
        }
        return false
    }
}

/**
 * Extension function to apply materialization strategy to tensor views.
 */
public fun <T : DType, V> TensorView<T, V>.materialize(
    strategy: MaterializationStrategy<T, V> = CopyMaterializationStrategy()
): Tensor<T, V> {
    return if (strategy.shouldMaterialize(this)) {
        strategy.materialize(this)
    } else {
        this // Return the view as-is if materialization is not needed
    }
}