package sk.ainet.core.tensor

import sk.ainet.core.tensor.dsl.TensorViewBuilder

/**
 * Extension functions for tensor slicing and view creation.
 * 
 * This file provides convenient methods for creating zero-copy tensor views
 * using various slicing patterns and optimizations.
 */

/**
 * Creates a zero-copy view of this tensor using slice descriptors.
 * 
 * @param descriptors List of slice descriptors defining the view
 * @return A SlicedTensorView that provides a view into this tensor
 */
public fun <T : DType, V> Tensor<T, V>.sliceView(descriptors: List<SliceDescriptor>): SlicedTensorView<T, V> {
    val indexMapper = SliceIndexMapper(this.shape, descriptors)
    return SlicedTensorView(this, indexMapper, descriptors)
}

/**
 * Creates a zero-copy view of this tensor using the DSL builder.
 * 
 * Usage:
 * val view = tensor.sliceView {
 *     segment { range(0, 5) }
 *     segment { all() }
 * }
 * 
 * @param builder DSL builder function for defining the view
 * @return A TensorView that provides a zero-copy view into this tensor
 */
public fun <T : DType, V> Tensor<T, V>.sliceView(builder: TensorViewBuilder<T, V>.() -> Unit): TensorView<T, V> {
    val viewBuilder = TensorViewBuilder(this)
    viewBuilder.builder()
    return viewBuilder.buildView()
}

/**
 * Intelligent slicing function that automatically chooses between zero-copy views and data copying
 * based on efficiency analysis.
 * 
 * @param builder DSL builder function for defining the slice
 * @param forceView If true, always create a view (may be slower for complex patterns)
 * @param forceCopy If true, always copy data (uses more memory but may be faster)
 * @return Either a TensorView or a new Tensor depending on the efficiency analysis
 */
public fun <T : DType, V> Tensor<T, V>.smartSlice(
    forceView: Boolean = false,
    forceCopy: Boolean = false,
    builder: TensorViewBuilder<T, V>.() -> Unit
): Tensor<T, V> {
    require(!(forceView && forceCopy)) { "Cannot force both view and copy simultaneously" }
    
    if (forceCopy) {
        // Create view first, then materialize it
        val view = sliceView(builder)
        return materializeView(view)
    }
    
    if (forceView) {
        return sliceView(builder)
    }
    
    // Intelligent decision based on efficiency analysis
    val view = sliceView(builder)
    return if (shouldUseView(view)) {
        view
    } else {
        materializeView(view)
    }
}

/**
 * Determines whether a view should be used based on efficiency analysis.
 * 
 * @param view The tensor view to analyze
 * @return true if the view is efficient, false if copying would be better
 */
private fun <T : DType, V> shouldUseView(view: TensorView<T, V>): Boolean {
    // Use view if it's contiguous (most efficient)
    if (view.isContiguous) {
        return true
    }
    
    // Calculate view complexity score
    val complexityScore = calculateViewComplexity(view)
    
    // Use view if complexity is low (arbitrary threshold of 10)
    return complexityScore <= 10
}

/**
 * Calculates a complexity score for the view access pattern.
 * Higher scores indicate more complex, potentially slower access patterns.
 * 
 * @param view The tensor view to analyze
 * @return Complexity score (lower is better)
 */
private fun <T : DType, V> calculateViewComplexity(view: TensorView<T, V>): Int {
    var complexity = 0
    
    // Add complexity for non-unit strides
    view.strides.forEach { stride ->
        if (stride != 1) {
            complexity += 2
        }
    }
    
    // Add complexity for dimension reductions
    if (view.shape.rank < view.parentTensor.shape.rank) {
        complexity += (view.parentTensor.shape.rank - view.shape.rank)
    }
    
    // Add complexity for small tensors (copying might be faster)
    val totalElements = view.shape.volume
    if (totalElements < 1000) { // Arbitrary small tensor threshold
        complexity += 5
    }
    
    return complexity
}

/**
 * Materializes a tensor view by copying its data into a new tensor.
 * This implementation provides basic materialization functionality.
 * 
 * @param view The view to materialize
 * @return A new tensor containing the view's data
 */
private fun <T : DType, V> materializeView(view: TensorView<T, V>): Tensor<T, V> {
    // Use the view's own materialize method, which provides the best available
    // materialization strategy for the specific view type
    val materializedData = view.materialize()
    
    // If the materialized data is already a full tensor, return it
    if (materializedData is Tensor<T, V>) {
        return materializedData
    }
    
    // For now, return the view itself as a fallback
    // In a complete implementation with tensor factories, this would:
    // 1. Determine the appropriate backend from the parent tensor
    // 2. Create a new concrete tensor using the backend's factory
    // 3. Copy all view data to the new tensor
    // 4. Return the new concrete tensor
    return view
}

/**
 * Analyzes slice patterns to detect contiguity and memory access efficiency.
 */
public object SlicePatternAnalyzer {
    
    /**
     * Determines if a slice pattern results in contiguous memory access.
     * 
     * @param descriptors The slice descriptors to analyze
     * @param parentShape The shape of the parent tensor
     * @return true if the slice pattern is contiguous
     */
    public fun isContiguousPattern(descriptors: List<SliceDescriptor>, parentShape: Shape): Boolean {
        // A pattern is contiguous if:
        // 1. All trailing dimensions use All or Range starting from 0
        // 2. No dimension reordering occurs
        // 3. No gaps in the access pattern
        
        var foundNonContiguous = false
        
        for (i in descriptors.indices.reversed()) {
            when (val desc = descriptors[i]) {
                is SliceDescriptor.All -> {
                    // All slices are contiguous
                    continue
                }
                is SliceDescriptor.Range -> {
                    // Range is contiguous if it starts from 0 and has step 1
                    if (desc.start != 0 || desc.step != 1) {
                        foundNonContiguous = true
                    }
                    // Once we find a non-contiguous range, all preceding dimensions must be single indices
                    if (foundNonContiguous) {
                        return false
                    }
                }
                is SliceDescriptor.Index -> {
                    // Index selections can maintain contiguity if they're in trailing dimensions
                    foundNonContiguous = true
                }
            }
        }
        
        return true
    }
    
    /**
     * Calculates the memory access pattern efficiency score.
     * Lower scores indicate more efficient access patterns.
     * 
     * @param descriptors The slice descriptors to analyze
     * @param parentShape The shape of the parent tensor
     * @return Efficiency score (0 = most efficient, higher = less efficient)
     */
    public fun calculateAccessPatternScore(descriptors: List<SliceDescriptor>, parentShape: Shape): Int {
        var score = 0
        
        for (i in descriptors.indices) {
            when (val desc = descriptors[i]) {
                is SliceDescriptor.All -> {
                    // All slices are most efficient (score 0)
                    score += 0
                }
                is SliceDescriptor.Range -> {
                    // Range efficiency depends on start position and step
                    if (desc.start != 0) score += 2
                    if (desc.step != 1) score += 3
                    
                    // Non-contiguous ranges in early dimensions are more expensive
                    val dimensionWeight = parentShape.rank - i
                    if (desc.start != 0 || desc.step != 1) {
                        score += dimensionWeight
                    }
                }
                is SliceDescriptor.Index -> {
                    // Index selections add moderate cost
                    score += 1
                    
                    // Index selections in early dimensions are more expensive
                    val dimensionWeight = parentShape.rank - i
                    score += dimensionWeight / 2
                }
            }
        }
        
        return score
    }
    
    /**
     * Analyzes if a slice pattern benefits from NCHW optimization.
     * 
     * @param descriptors The slice descriptors to analyze (must be 4D for NCHW)
     * @return true if NCHW optimization would be beneficial
     */
    public fun benefitsFromNCHWOptimization(descriptors: List<SliceDescriptor>): Boolean {
        if (descriptors.size != 4) return false
        
        // NCHW optimization benefits cases like:
        // 1. Channel extraction: [:, c, :, :] 
        // 2. Batch slicing: [n1:n2, :, :, :]
        // 3. Spatial slicing: [:, :, h1:h2, w1:w2]
        
        val batchDesc = descriptors[0]
        val channelDesc = descriptors[1] 
        val heightDesc = descriptors[2]
        val widthDesc = descriptors[3]
        
        // Channel extraction optimization
        if (channelDesc is SliceDescriptor.Index && 
            batchDesc is SliceDescriptor.All &&
            heightDesc is SliceDescriptor.All && 
            widthDesc is SliceDescriptor.All) {
            return true
        }
        
        // Batch slicing optimization
        if (batchDesc is SliceDescriptor.Range &&
            channelDesc is SliceDescriptor.All &&
            heightDesc is SliceDescriptor.All && 
            widthDesc is SliceDescriptor.All) {
            return true
        }
        
        // Spatial slicing optimization
        if (batchDesc is SliceDescriptor.All &&
            channelDesc is SliceDescriptor.All &&
            (heightDesc is SliceDescriptor.Range || widthDesc is SliceDescriptor.Range)) {
            return true
        }
        
        return false
    }
}

/**
 * Creates a zero-copy view of this tensor using NCHW-optimized mapping.
 * 
 * @param viewOffset The offset in each NCHW dimension
 * @param viewStrides The stride in each NCHW dimension
 * @return A SlicedTensorView optimized for NCHW access patterns
 */
public fun <T : DType, V> Tensor<T, V>.nchwView(
    viewOffset: IntArray,
    viewStrides: IntArray
): SlicedTensorView<T, V> {
    require(this.shape.rank == 4) { "NCHW views require 4D tensors" }
    
    val indexMapper = NCHWIndexMapper(this.shape, viewOffset, viewStrides)
    
    // Convert NCHW parameters to slice descriptors for consistency
    val descriptors = buildList<SliceDescriptor> {
        for (i in 0..3) {
            if (viewStrides[i] == 0) {
                add(SliceDescriptor.Index(viewOffset[i]))
            } else if (viewStrides[i] == 1 && viewOffset[i] == 0) {
                add(SliceDescriptor.All)
            } else {
                val end = viewOffset[i] + (this@nchwView.shape[i] - viewOffset[i]) / viewStrides[i]
                add(SliceDescriptor.Range(viewOffset[i], end, viewStrides[i]))
            }
        }
    }
    
    return SlicedTensorView(this, indexMapper, descriptors)
}

/**
 * Creates a batch slice view: tensor[startBatch:endBatch, :, :, :]
 */
public fun <T : DType, V> Tensor<T, V>.batchSlice(
    startBatch: Int, 
    endBatch: Int, 
    step: Int = 1
): SlicedTensorView<T, V> {
    require(this.shape.rank == 4) { "Batch slicing requires 4D tensors (NCHW format)" }
    
    return nchwView(
        viewOffset = intArrayOf(startBatch, 0, 0, 0),
        viewStrides = intArrayOf(step, 1, 1, 1)
    )
}

/**
 * Creates a channel extraction view: tensor[:, channelIndex, :, :]
 */
public fun <T : DType, V> Tensor<T, V>.channelSlice(channelIndex: Int): SlicedTensorView<T, V> {
    require(this.shape.rank == 4) { "Channel slicing requires 4D tensors (NCHW format)" }
    require(channelIndex >= 0 && channelIndex < this.shape[1]) { 
        "Channel index $channelIndex out of bounds (channels: ${this.shape[1]})" 
    }
    
    return nchwView(
        viewOffset = intArrayOf(0, channelIndex, 0, 0),
        viewStrides = intArrayOf(1, 0, 1, 1)  // 0 stride collapses the dimension
    )
}

/**
 * Creates a spatial region view: tensor[:, :, h1:h2, w1:w2]
 */
public fun <T : DType, V> Tensor<T, V>.spatialSlice(
    heightStart: Int, heightEnd: Int, heightStep: Int = 1,
    widthStart: Int, widthEnd: Int, widthStep: Int = 1
): SlicedTensorView<T, V> {
    require(this.shape.rank == 4) { "Spatial slicing requires 4D tensors (NCHW format)" }
    
    return nchwView(
        viewOffset = intArrayOf(0, 0, heightStart, widthStart),
        viewStrides = intArrayOf(1, 1, heightStep, widthStep)
    )
}

/**
 * Creates a general slice view using range notation.
 * 
 * @param ranges Vararg of ranges for each dimension. Use null to select all elements in a dimension.
 */
public fun <T : DType, V> Tensor<T, V>.slice(vararg ranges: IntRange?): SlicedTensorView<T, V> {
    require(ranges.size <= this.shape.rank) { 
        "Too many slice ranges: ${ranges.size} > tensor rank ${this.shape.rank}" 
    }
    
    val descriptors = buildList<SliceDescriptor> {
        for (i in ranges.indices) {
            val range = ranges[i]
            if (range == null) {
                add(SliceDescriptor.All)
            } else {
                add(SliceDescriptor.Range(range.first, range.last + 1))  // +1 because IntRange is inclusive
            }
        }
        
        // Fill remaining dimensions with All
        repeat(this@slice.shape.rank - ranges.size) {
            add(SliceDescriptor.All)
        }
    }
    
    return sliceView(descriptors)
}

/**
 * Creates a view by selecting specific indices in each dimension.
 * Use null to select all elements in a dimension.
 */
public fun <T : DType, V> Tensor<T, V>.at(vararg indices: Int?): SlicedTensorView<T, V> {
    require(indices.size <= this.shape.rank) { 
        "Too many indices: ${indices.size} > tensor rank ${this.shape.rank}" 
    }
    
    val descriptors = buildList<SliceDescriptor> {
        for (i in indices.indices) {
            val index = indices[i]
            if (index == null) {
                add(SliceDescriptor.All)
            } else {
                require(index >= 0 && index < this@at.shape[i]) {
                    "Index $index out of bounds for dimension $i (size: ${this@at.shape[i]})"
                }
                add(SliceDescriptor.Index(index))
            }
        }
        
        // Fill remaining dimensions with All
        repeat(this@at.shape.rank - indices.size) {
            add(SliceDescriptor.All)
        }
    }
    
    return sliceView(descriptors)
}