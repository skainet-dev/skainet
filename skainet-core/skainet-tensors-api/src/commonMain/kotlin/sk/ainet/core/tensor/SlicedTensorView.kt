package sk.ainet.core.tensor

/**
 * Implementation of a zero-copy sliced tensor view.
 * 
 * SlicedTensorView provides a view into a parent tensor without copying data.
 * It supports lazy shape computation, efficient index mapping, and view chaining
 * (creating views of views).
 * 
 * @param T The data type of the tensor elements
 * @param V The value type of the tensor elements
 * @param parentTensor The parent tensor this view references
 * @param indexMapping The index mapper for coordinate transformation
 * @param sliceDescriptors The slice descriptors that define this view
 */
public class SlicedTensorView<T : DType, V>(
    override val parentTensor: Tensor<T, V>,
    override val indexMapping: IndexMapper,
    private val sliceDescriptors: List<SliceDescriptor>
) : TensorView<T, V> {
    
    // Delegate TensorOps methods to parent tensor
    override fun matmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> = parentTensor.matmul(a, b)
    override fun matmul4d(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> = parentTensor.matmul4d(a, b)
    override fun scale(a: Tensor<T, V>, scalar: Double): Tensor<T, V> = parentTensor.scale(a, scalar)
    override fun dot(a: Tensor<T, V>, b: Tensor<T, V>): Double = parentTensor.dot(a, b)
    
    // Tensor-Tensor operations
    override fun Tensor<T, V>.plus(other: Tensor<T, V>): Tensor<T, V> = with(parentTensor) { this@plus.plus(other) }
    override fun Tensor<T, V>.minus(other: Tensor<T, V>): Tensor<T, V> = with(parentTensor) { this@minus.minus(other) }
    override fun Tensor<T, V>.times(other: Tensor<T, V>): Tensor<T, V> = with(parentTensor) { this@times.times(other) }
    override fun Tensor<T, V>.div(other: Tensor<T, V>): Tensor<T, V> = with(parentTensor) { this@div.div(other) }
    
    // Tensor-Scalar operations
    override fun Tensor<T, V>.plus(scalar: Int): Tensor<T, V> = with(parentTensor) { this@plus.plus(scalar) }
    override fun Tensor<T, V>.minus(scalar: Int): Tensor<T, V> = with(parentTensor) { this@minus.minus(scalar) }
    override fun Tensor<T, V>.times(scalar: Int): Tensor<T, V> = with(parentTensor) { this@times.times(scalar) }
    override fun Tensor<T, V>.div(scalar: Int): Tensor<T, V> = with(parentTensor) { this@div.div(scalar) }
    
    override fun Tensor<T, V>.plus(scalar: Float): Tensor<T, V> = with(parentTensor) { this@plus.plus(scalar) }
    override fun Tensor<T, V>.minus(scalar: Float): Tensor<T, V> = with(parentTensor) { this@minus.minus(scalar) }
    override fun Tensor<T, V>.times(scalar: Float): Tensor<T, V> = with(parentTensor) { this@times.times(scalar) }
    override fun Tensor<T, V>.div(scalar: Float): Tensor<T, V> = with(parentTensor) { this@div.div(scalar) }
    
    override fun Tensor<T, V>.plus(scalar: Double): Tensor<T, V> = with(parentTensor) { this@plus.plus(scalar) }
    override fun Tensor<T, V>.minus(scalar: Double): Tensor<T, V> = with(parentTensor) { this@minus.minus(scalar) }
    override fun Tensor<T, V>.times(scalar: Double): Tensor<T, V> = with(parentTensor) { this@times.times(scalar) }
    override fun Tensor<T, V>.div(scalar: Double): Tensor<T, V> = with(parentTensor) { this@div.div(scalar) }
    
    // Scalar-Tensor operations
    override fun Double.plus(t: Tensor<T, V>): Tensor<T, V> = with(parentTensor) { this@plus.plus(t) }
    override fun Double.minus(t: Tensor<T, V>): Tensor<T, V> = with(parentTensor) { this@minus.minus(t) }
    override fun Double.times(t: Tensor<T, V>): Tensor<T, V> = with(parentTensor) { this@times.times(t) }
    override fun Double.div(t: Tensor<T, V>): Tensor<T, V> = with(parentTensor) { this@div.div(t) }
    
    // Activation and transformation functions
    override fun Tensor<T, V>.t(): Tensor<T, V> = with(parentTensor) { this@t.t() }
    override fun Tensor<T, V>.relu(): Tensor<T, V> = with(parentTensor) { this@relu.relu() }
    override fun Tensor<T, V>.softmax(dimension: Int): Tensor<T, V> = with(parentTensor) { this@softmax.softmax(dimension) }
    override fun Tensor<T, V>.sigmoid(): Tensor<T, V> = with(parentTensor) { this@sigmoid.sigmoid() }
    override fun Tensor<T, V>.tanh(): Tensor<T, V> = with(parentTensor) { this@tanh.tanh() }
    override fun Tensor<T, V>.flatten(startDim: Int, endDim: Int): Tensor<T, V> = with(parentTensor) { this@flatten.flatten(startDim, endDim) }
    
    // Lazy initialization of view shape
    override val viewShape: Shape by lazy {
        computeViewShape(parentTensor.shape, sliceDescriptors)
    }
    
    // Cached stride information for performance
    override val strides: IntArray by lazy {
        computeViewStrides(parentTensor.shape, sliceDescriptors)
    }
    
    // Cached contiguity check for optimization decisions
    override val isContiguous: Boolean by lazy {
        computeContiguity(sliceDescriptors)
    }
    
    /**
     * Efficiently retrieves the value at the specified indices using cached metadata.
     * 
     * @param indices The indices in the view coordinate system
     * @return The value at the specified indices
     */
    override operator fun get(vararg indices: Int): V {
        // Validate indices against view shape
        require(indices.size == viewShape.rank) {
            "Expected ${viewShape.rank} indices, got ${indices.size}"
        }
        
        // Bounds checking
        for (i in indices.indices) {
            require(indices[i] >= 0 && indices[i] < viewShape[i]) {
                "Index ${indices[i]} out of bounds for dimension $i (size: ${viewShape[i]})"
            }
        }
        
        // Use index mapping to get parent coordinates
        val parentIndices = indexMapping.mapToParent(indices)
        return parentTensor.get(*parentIndices)
    }
    
    /**
     * Creates a view of this view (chained views).
     * 
     * @param newSliceDescriptors Additional slice descriptors to apply
     * @return A new SlicedTensorView that chains with this view
     */
    public fun createSubView(newSliceDescriptors: List<SliceDescriptor>): SlicedTensorView<T, V> {
        // For views of views, we need to compose the slice descriptors
        val composedDescriptors = composeSliceDescriptors(sliceDescriptors, newSliceDescriptors)
        
        // Create new index mapper that chains the transformations
        val composedMapper = when (indexMapping) {
            is SliceIndexMapper -> SliceIndexMapper(parentTensor.shape, composedDescriptors)
            is NCHWIndexMapper -> {
                // For NCHW, we need to recompute the offsets and strides
                val (newOffset, newStrides) = computeComposedNCHWParameters(
                    indexMapping, newSliceDescriptors
                )
                NCHWIndexMapper(parentTensor.shape, newOffset, newStrides)
            }
            else -> SliceIndexMapper(parentTensor.shape, composedDescriptors)
        }
        
        return SlicedTensorView(parentTensor, composedMapper, composedDescriptors)
    }
    
    /**
     * Materializes this view into a new tensor by copying the data.
     * This is useful when the view becomes too complex or for performance reasons.
     */
    public override fun materialize(): TensorData<T, V> {
        // For now, return the view itself as materialization requires tensor factories
        // which are not yet integrated into the view system.
        // In a complete implementation, this would:
        // 1. Determine the appropriate tensor backend/factory based on the parent tensor
        // 2. Create a new concrete tensor with the view's shape
        // 3. Copy all view data using copyTo() method
        // 4. Return the new concrete tensor as TensorData
        return this
    }
    
    /**
     * Offset in the parent tensor data where this view begins.
     */
    override val offset: Int get() = 0  // Default implementation for SlicedTensorView
    
    /**
     * Copies the view data to the destination array.
     */
    override fun copyTo(dest: Array<V>, destOffset: Int) {
        var index = destOffset
        // Iterate through all elements in the view using its shape
        val totalElements = viewShape.volume
        for (i in 0 until totalElements) {
            val indices = flatIndexToMultiIndex(i, viewShape)
            dest[index++] = this.get(*indices)
        }
    }
    
    /**
     * Converts a flat index to multi-dimensional indices for the given shape.
     */
    private fun flatIndexToMultiIndex(flatIndex: Int, shape: Shape): IntArray {
        val indices = IntArray(shape.dimensions.size)
        var remaining = flatIndex
        
        for (i in shape.dimensions.size - 1 downTo 0) {
            val dimSize = shape.dimensions[i]
            indices[i] = remaining % dimSize
            remaining /= dimSize
        }
        
        return indices
    }
    
    /**
     * Creates a slice view of this tensor view.
     */
    override fun slice(ranges: IntArray): TensorData<T, V> {
        require(ranges.size == viewShape.rank * 2) {
            "Ranges array must have size ${viewShape.rank * 2} (start,end pairs), got ${ranges.size}"
        }
        
        // Convert ranges to SliceDescriptors
        val newSliceDescriptors = mutableListOf<SliceDescriptor>()
        for (i in 0 until viewShape.rank) {
            val start = ranges[i * 2]
            val end = ranges[i * 2 + 1]
            val dimSize = viewShape.dimensions[i]
            
            require(start >= 0 && start < dimSize) {
                "Start index $start out of bounds for view dimension $i (size $dimSize)"
            }
            require(end > start && end <= dimSize) {
                "End index $end must be > start ($start) and <= view dimension size ($dimSize)"
            }
            
            newSliceDescriptors.add(SliceDescriptor.Range(start, end, 1))
        }
        
        // Create a sub-view that combines this view's slicing with the new slicing
        return createSubView(newSliceDescriptors)
    }
    
    private fun computeViewShape(parentShape: Shape, descriptors: List<SliceDescriptor>): Shape {
        val viewDimensions = mutableListOf<Int>()
        
        for (i in descriptors.indices) {
            val descriptor = descriptors[i]
            val parentDimSize = parentShape[i]
            
            when (descriptor) {
                is SliceDescriptor.Range -> {
                    val actualEnd = minOf(descriptor.end, parentDimSize)
                    val actualStart = maxOf(descriptor.start, 0)
                    val size = maxOf(0, (actualEnd - actualStart + descriptor.step - 1) / descriptor.step)
                    viewDimensions.add(size)
                }
                is SliceDescriptor.Index -> {
                    // Index selection reduces dimension - don't add to view dimensions
                }
                is SliceDescriptor.All -> {
                    viewDimensions.add(parentDimSize)
                }
            }
        }
        
        return Shape(viewDimensions.toIntArray())
    }
    
    private fun computeViewStrides(parentShape: Shape, descriptors: List<SliceDescriptor>): IntArray {
        // Compute row-major strides for the view
        val viewDims = viewShape.dimensions
        val strides = IntArray(viewDims.size)
        
        if (viewDims.isNotEmpty()) {
            strides[viewDims.size - 1] = 1
            for (i in viewDims.size - 2 downTo 0) {
                strides[i] = strides[i + 1] * viewDims[i + 1]
            }
        }
        
        return strides
    }
    
    private fun computeContiguity(descriptors: List<SliceDescriptor>): Boolean {
        // A view is contiguous if all slices are either All or have step=1
        // and there are no interspersed Index selections that break continuity
        var lastWasIndex = false
        
        for (descriptor in descriptors) {
            when (descriptor) {
                is SliceDescriptor.Range -> {
                    if (descriptor.step != 1) return false
                    lastWasIndex = false
                }
                is SliceDescriptor.Index -> {
                    if (lastWasIndex) return false  // Multiple index selections break contiguity
                    lastWasIndex = true
                }
                is SliceDescriptor.All -> {
                    lastWasIndex = false
                }
            }
        }
        
        return true
    }
    
    private fun composeSliceDescriptors(
        existing: List<SliceDescriptor>,
        additional: List<SliceDescriptor>
    ): List<SliceDescriptor> {
        // Compose two sets of slice descriptors
        // This is a simplified implementation - a full version would handle all cases
        val composed = mutableListOf<SliceDescriptor>()
        var existingIndex = 0
        
        for (newDescriptor in additional) {
            when (newDescriptor) {
                is SliceDescriptor.Range -> {
                    // Apply range to existing range/all
                    while (existingIndex < existing.size && existing[existingIndex] is SliceDescriptor.Index) {
                        composed.add(existing[existingIndex])
                        existingIndex++
                    }
                    
                    if (existingIndex < existing.size) {
                        val existingDesc = existing[existingIndex]
                        when (existingDesc) {
                            is SliceDescriptor.Range -> {
                                val newStart = existingDesc.start + newDescriptor.start * existingDesc.step
                                val newStep = existingDesc.step * newDescriptor.step
                                composed.add(SliceDescriptor.Range(newStart, newStart + newDescriptor.end * newStep, newStep))
                            }
                            is SliceDescriptor.All -> {
                                composed.add(newDescriptor)
                            }
                            is SliceDescriptor.Index -> {
                                composed.add(existingDesc)
                            }
                        }
                        existingIndex++
                    }
                }
                is SliceDescriptor.Index -> {
                    composed.add(newDescriptor)
                }
                is SliceDescriptor.All -> {
                    if (existingIndex < existing.size) {
                        composed.add(existing[existingIndex])
                        existingIndex++
                    }
                }
            }
        }
        
        // Add remaining existing descriptors
        while (existingIndex < existing.size) {
            composed.add(existing[existingIndex])
            existingIndex++
        }
        
        return composed
    }
    
    private fun computeComposedNCHWParameters(
        existingMapper: NCHWIndexMapper,
        newDescriptors: List<SliceDescriptor>
    ): Pair<IntArray, IntArray> {
        // Placeholder for NCHW composition - would need access to existing mapper internals
        return Pair(intArrayOf(0, 0, 0, 0), intArrayOf(1, 1, 1, 1))
    }
}