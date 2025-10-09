package sk.ainet.lang.tensor

import sk.ainet.lang.types.DType

/**
 * Extension methods for tensor view materialization.
 * 
 * These extensions provide convenient access to different materialization
 * strategies, allowing tensor views to be converted into standalone tensors
 * with various trade-offs between memory usage and performance.
 */

/**
 * Materializes this tensor view into a standalone tensor using the default strategy.
 * 
 * This method creates a new tensor that contains a copy of all data from the view,
 * making it independent of the parent tensor. The default strategy uses immediate
 * copying (CopyMaterializationStrategy) which provides predictable performance
 * and memory usage characteristics.
 * 
 * ## Usage Examples
 * 
 * ```kotlin
 * val parentTensor = tensorOf(shape = Shape(4, 4)) { /* initialization */ }
 * val view = parentTensor.sliceView { range(1, 3) }
 * val materialized = view.materialize() // Independent copy of the view data
 * ```
 * 
 * ## Materialization Benefits
 * 
 * - **Independence**: Result tensor has no dependency on parent tensor
 * - **Performance**: Direct memory access without coordinate transformation
 * - **Compatibility**: Works with all tensor operations and external libraries
 * - **Memory Management**: Allows parent tensor to be garbage collected
 * 
 * @param T the data type constraint extending DType
 * @param V the actual value type that will be stored and accessed
 * @return a new Tensor containing a materialized copy of this view's data
 * @throws IllegalStateException if the view cannot be materialized
 */
public fun <T : DType, V> TensorView<T, V>.materialize(): Tensor<T, V> {
    val strategy = CopyMaterializationStrategy<T, V>()
    return strategy.materialize(this)
}

/**
 * Materializes this tensor view using a specific materialization strategy.
 * 
 * This method allows explicit control over the materialization process by
 * selecting a specific strategy. Different strategies provide different
 * trade-offs between memory usage, performance, and access patterns.
 * 
 * ## Available Strategies
 * 
 * - **CopyMaterializationStrategy**: Immediate copying with full memory allocation
 * - **LazyMaterializationStrategy**: Deferred copying with sparse element caching
 * 
 * ## Usage Examples
 * 
 * ```kotlin
 * val view = parentTensor.sliceView { /* slicing definition */ }
 * 
 * // Use lazy materialization for sparse access patterns
 * val lazy = view.materialize(LazyMaterializationStrategy())
 * 
 * // Use copy materialization for frequent access
 * val copy = view.materialize(CopyMaterializationStrategy())
 * ```
 * 
 * @param T the data type constraint extending DType
 * @param V the actual value type that will be stored and accessed
 * @param strategy the MaterializationStrategy to use for materialization
 * @return a new Tensor created using the specified materialization strategy
 * @throws IllegalStateException if the strategy cannot materialize this view
 */
public fun <T : DType, V> TensorView<T, V>.materialize(
    strategy: MaterializationStrategy<T, V>
): Tensor<T, V> {
    require(strategy.canMaterialize(this)) {
        "Strategy '${strategy.name}' cannot materialize this view"
    }
    return strategy.materialize(this)
}

/**
 * Checks if this tensor view can be materialized using the default strategy.
 * 
 * This method provides a quick check to determine if materialization is
 * possible without actually performing the materialization operation.
 * Useful for defensive programming and resource planning.
 * 
 * @param T the data type constraint extending DType
 * @param V the actual value type that will be stored and accessed
 * @return true if this view can be materialized, false otherwise
 */
public fun <T : DType, V> TensorView<T, V>.canMaterialize(): Boolean {
    val strategy = CopyMaterializationStrategy<T, V>()
    return strategy.canMaterialize(this)
}

/**
 * Checks if this tensor view can be materialized using a specific strategy.
 * 
 * This method allows checking materialization compatibility with a specific
 * strategy before attempting the actual materialization operation.
 * 
 * @param T the data type constraint extending DType
 * @param V the actual value type that will be stored and accessed
 * @param strategy the MaterializationStrategy to check compatibility with
 * @return true if the strategy can materialize this view, false otherwise
 */
public fun <T : DType, V> TensorView<T, V>.canMaterialize(
    strategy: MaterializationStrategy<T, V>
): Boolean {
    return strategy.canMaterialize(this)
}

/**
 * Estimates the memory overhead of materializing this tensor view.
 * 
 * This method provides insight into the memory cost of materialization
 * using the default strategy, helping with resource planning and
 * memory management decisions.
 * 
 * @param T the data type constraint extending DType
 * @param V the actual value type that will be stored and accessed
 * @return estimated memory overhead in bytes for default materialization
 */
public fun <T : DType, V> TensorView<T, V>.estimateMaterializationCost(): Long {
    val strategy = CopyMaterializationStrategy<T, V>()
    return strategy.estimateMemoryOverhead(this)
}

/**
 * Estimates the memory overhead of materializing this tensor view with a specific strategy.
 * 
 * This method provides strategy-specific memory cost estimation,
 * allowing comparison between different materialization approaches.
 * 
 * @param T the data type constraint extending DType
 * @param V the actual value type that will be stored and accessed
 * @param strategy the MaterializationStrategy to estimate costs for
 * @return estimated memory overhead in bytes for the specified strategy
 */
public fun <T : DType, V> TensorView<T, V>.estimateMaterializationCost(
    strategy: MaterializationStrategy<T, V>
): Long {
    return strategy.estimateMemoryOverhead(this)
}