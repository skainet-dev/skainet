package sk.ainet.core.tensor

/**
 * Interface representing a zero-copy view of a tensor.
 * 
 * A TensorView provides a different perspective on existing tensor data without copying it.
 * It maintains a reference to the parent tensor and maps indices from the view coordinate
 * system to the parent coordinate system.
 * 
 * @param T The data type of the tensor elements
 * @param V The value type of the tensor elements
 */
public interface TensorView<T : DType, V> : Tensor<T, V> {
    
    /**
     * The parent tensor that this view references.
     */
    public val parentTensor: Tensor<T, V>
    
    /**
     * The shape of this view, which may differ from the parent tensor's shape.
     */
    public val viewShape: Shape
    
    /**
     * The index mapper that converts view indices to parent tensor indices.
     */
    public val indexMapping: IndexMapper
    
    /**
     * Whether this view represents contiguous data in memory.
     * Contiguous views can be optimized for better performance.
     */
    public override val isContiguous: Boolean
    
    /**
     * Stride information for each dimension in the view.
     * Represents the step size in the parent tensor for each dimension.
     */
    public override val strides: IntArray
    
    /**
     * Offset in the parent tensor data where this view begins.
     */
    public override val offset: Int get() = 0  // Default implementation
    
    /**
     * Retrieves the value at the specified indices in the view coordinate system.
     * The indices are mapped to the parent tensor's coordinate system before access.
     *
     * @param indices The indices in the view coordinate system
     * @return The value at the specified indices
     */
    override operator fun get(vararg indices: Int): V {
        val parentIndices = indexMapping.mapToParent(indices)
        return parentTensor.get(*parentIndices)
    }
    
    /**
     * Returns the shape of this view.
     * This is an alias for viewShape to implement the Tensor interface.
     */
    override val shape: Shape
        get() = viewShape
    
    /**
     * Default implementation of copyTo for tensor views.
     * Subclasses should override for better performance.
     */
    override fun copyTo(dest: Array<V>, destOffset: Int) {
        var index = destOffset
        val totalElements = viewShape.volume
        
        // Iterate through all elements in the view using its shape
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
     * Default implementation of slice for tensor views.
     */
    override fun slice(ranges: IntArray): TensorData<T, V> {
        require(ranges.size == viewShape.rank * 2) {
            "Ranges array must have size ${viewShape.rank * 2} (start,end pairs), got ${ranges.size}"
        }
        
        // Convert ranges to SliceDescriptors
        val sliceDescriptors = mutableListOf<SliceDescriptor>()
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
            
            sliceDescriptors.add(SliceDescriptor.Range(start, end, 1))
        }
        
        // Create a new SlicedTensorView that combines existing view with new slice
        return if (this is SlicedTensorView) {
            // For already sliced views, create a sub-view
            this.createSubView(sliceDescriptors)
        } else {
            // For base views, create new SlicedTensorView
            val indexMapper = SliceIndexMapper(parentTensor.shape, sliceDescriptors)
            SlicedTensorView(parentTensor, indexMapper, sliceDescriptors)
        }
    }
    
    /**
     * Default implementation of materialize for tensor views.
     */
    override fun materialize(): TensorData<T, V> {
        // For now, return the view itself as materialization requires tensor factories
        // which are not yet integrated into the view system.
        // In a complete implementation, this would:
        // 1. Create a new concrete tensor using the appropriate factory
        // 2. Copy all view data into the new tensor
        // 3. Return the new tensor as TensorData
        return this
    }
}