package sk.ainet.lang.tensor

import sk.ainet.lang.types.DType

/**
 * DSL builder for creating tensor slices with fluent syntax.
 *
 * This class provides a DSL for defining tensor slicing operations using
 * a fluent interface. It supports various slicing patterns including:
 * - Range selection: `range(start, end)`
 * - Single index selection: `at(index)`
 * - Full dimension selection: `all()`
 * - Strided access: `step(start, end, step)`
 *
 * The builder is designed to work with the segment { } block pattern
 * where each call to segment defines slicing for one dimension.
 *
 * ## Usage Example
 * ```kotlin
 * tensor.sliceView {
 *     segment { range(0, 10) }  // First dimension: select indices 0-9
 *     segment { at(5) }         // Second dimension: select index 5
 *     segment { all() }         // Third dimension: select all indices
 *     segment { step(0, 20, 2) } // Fourth dimension: select every 2nd index from 0-19
 * }
 * ```
 *
 * @param T the data type constraint extending DType
 * @param V the actual value type
 */
public class TensorSliceBuilder<T : DType, V> {
    private val slices = mutableListOf<Slice<T, V>>()
    
    /**
     * Current segment being built. Used internally to track which dimension
     * is being configured in the current segment { } block.
     */
    private var currentSegmentIndex = 0
    
    /**
     * Defines slicing for the next dimension using the provided block.
     *
     * Each call to segment configures slicing for one dimension of the tensor.
     * The dimensions are configured in order, starting from dimension 0.
     *
     * @param block the configuration block that defines the slice for this dimension
     */
    public fun segment(block: SegmentBuilder<T, V>.() -> Slice<T, V>) {
        val builder = SegmentBuilder<T, V>(currentSegmentIndex)
        val slice = builder.block()
        slices.add(slice)
        currentSegmentIndex++
    }
    
    /**
     * Builds the final list of slices configured by this builder.
     *
     * @return the list of slice descriptors for each dimension
     */
    internal fun build(): List<Slice<T, V>> = slices.toList()
    
    /**
     * Validates that the slice configuration is complete and consistent.
     *
     * @param tensorShape the shape of the tensor being sliced
     * @throws IllegalStateException if the slice configuration is invalid
     */
    internal fun validate(tensorShape: Shape) {
        if (slices.size != tensorShape.rank) {
            throw IllegalStateException(
                "Slice configuration incomplete: expected ${tensorShape.rank} dimensions, got ${slices.size}"
            )
        }
        
        slices.forEachIndexed { index, slice ->
            val dimensionSize = tensorShape[index]
            if (!slice.isValid(dimensionSize)) {
                throw IllegalStateException(
                    "Invalid slice at dimension $index: $slice (dimension size: $dimensionSize)"
                )
            }
        }
    }
}

/**
 * Builder for configuring a single dimension slice within a segment { } block.
 *
 * This class provides the methods available within each segment { } block
 * for defining how that particular dimension should be sliced.
 *
 * @param T the data type constraint extending DType
 * @param V the actual value type
 * @param dimensionIndex the index of the dimension being configured (for error reporting)
 */
public class SegmentBuilder<T : DType, V>(private val dimensionIndex: Int) {
    
    /**
     * Selects a range of indices from start (inclusive) to end (exclusive).
     *
     * This is equivalent to Python's slice notation `start:end`.
     * Supports negative indexing where -1 represents the last element.
     *
     * @param start the starting index (inclusive), can be negative
     * @param end the ending index (exclusive), can be negative
     * @return a Range slice descriptor
     * @throws IllegalArgumentException if start >= end after normalization
     */
    public fun range(start: Int, end: Int): Slice<T, V> {
        // Note: negative index normalization will be handled during tensor slicing
        // when dimension size is known. Here we just validate the relationship.
        if (start >= 0 && end >= 0) {
            require(end > start) { "Range end must be greater than start: end=$end, start=$start" }
        }
        return Slice.Range(start, end)
    }
    
    /**
     * Selects a single index, reducing the dimensionality by 1.
     *
     * This is equivalent to Python's slice notation `[index]`.
     * Supports negative indexing where -1 represents the last element.
     *
     * @param index the index to select, can be negative
     * @return an At slice descriptor
     */
    public fun at(index: Int): Slice<T, V> {
        // Note: negative index normalization will be handled during tensor slicing
        return Slice.At(index)
    }
    
    /**
     * Selects all indices in this dimension.
     *
     * This is equivalent to Python's slice notation `:` or `[:]`.
     *
     * @return an All slice descriptor
     */
    public fun all(): Slice<T, V> = Slice.All()
    
    /**
     * Selects indices from start to end with a specified step size.
     *
     * This is equivalent to Python's slice notation `start:end:step`.
     * Supports negative indexing for start and end parameters.
     *
     * @param start the starting index (inclusive), can be negative
     * @param end the ending index (exclusive), can be negative
     * @param step the step size (must be positive)
     * @return a Step slice descriptor
     * @throws IllegalArgumentException if step <= 0
     */
    public fun step(start: Int, end: Int, step: Int): Slice<T, V> {
        require(step > 0) { "Step size must be positive: $step" }
        // Note: negative index normalization and range validation will be handled
        // during tensor slicing when dimension size is known
        return Slice.Step(start, end, step)
    }
    
    // Range operator support extensions
    
    /**
     * Creates a range slice using Kotlin's range operator.
     *
     * This enables syntax like: `segment { 0..10 }`
     *
     * @receiver the IntRange to convert to a slice
     * @return a Range slice descriptor
     */
    public operator fun IntRange.unaryPlus(): Slice<T, V> = range(first, last + 1)
    
    /**
     * Creates a range slice using Kotlin's until operator.
     *
     * This enables syntax like: `segment { 5 until 20 }`
     *
     * @receiver the IntRange to convert to a slice (exclusive end)
     * @return a Range slice descriptor
     */
    public operator fun IntRange.unaryMinus(): Slice<T, V> = range(first, last)
    
    /**
     * Validates slice bounds against dimension constraints.
     *
     * This method provides additional validation beyond the basic DSL methods
     * for ensuring slice consistency within the context of a specific dimension.
     *
     * @param slice the slice to validate
     * @param dimensionSize the size of the dimension (if known)
     * @return error message if validation fails, null if valid
     */
    internal fun validateSliceBounds(slice: Slice<T, V>, dimensionSize: Int?): String? {
        if (dimensionSize == null) return null
        
        return when (slice) {
            is Slice.Range<T, V> -> {
                val normalizedStart = if (slice.start < 0) dimensionSize + slice.start else slice.start
                val normalizedEnd = if (slice.end < 0) dimensionSize + slice.end else slice.end
                
                when {
                    normalizedStart < 0 -> "Start index out of bounds: ${slice.start} (normalized: $normalizedStart)"
                    normalizedEnd > dimensionSize -> "End index out of bounds: ${slice.end} (normalized: $normalizedEnd)"
                    normalizedStart >= normalizedEnd -> "Invalid range: start >= end after normalization ($normalizedStart >= $normalizedEnd)"
                    else -> null
                }
            }
            is Slice.At<T, V> -> {
                val normalizedIndex = if (slice.index < 0) dimensionSize + slice.index else slice.index
                when {
                    normalizedIndex < 0 || normalizedIndex >= dimensionSize -> 
                        "Index out of bounds: ${slice.index} (normalized: $normalizedIndex, dimension size: $dimensionSize)"
                    else -> null
                }
            }
            is Slice.Step<T, V> -> {
                val normalizedStart = if (slice.start < 0) dimensionSize + slice.start else slice.start
                val normalizedEnd = if (slice.end < 0) dimensionSize + slice.end else slice.end
                
                when {
                    normalizedStart < 0 -> "Step start index out of bounds: ${slice.start} (normalized: $normalizedStart)"
                    normalizedEnd > dimensionSize -> "Step end index out of bounds: ${slice.end} (normalized: $normalizedEnd)"
                    normalizedStart >= normalizedEnd -> "Invalid step range: start >= end after normalization ($normalizedStart >= $normalizedEnd)"
                    else -> null
                }
            }
            else -> null // All slices are always valid
        }
    }
}