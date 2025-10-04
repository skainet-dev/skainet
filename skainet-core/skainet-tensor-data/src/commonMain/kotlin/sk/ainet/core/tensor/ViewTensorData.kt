package sk.ainet.core.tensor

/**
 * View tensor data implementation for stride-based access.
 * This represents a view into existing tensor data without copying memory.
 * Views can represent slices, transposes, or other transformations of the parent data.
 */
public class ViewTensorData<T : DType, V>(
    private val parentData: Array<V>,
    override val shape: Shape,
    override val strides: IntArray,
    override val offset: Int,
    private val parentShape: Shape
) : TensorData<T, V> {

    override val isContiguous: Boolean by lazy {
        // Check if this view represents contiguous memory
        if (shape.dimensions.isEmpty()) return@lazy true
        
        val expectedStrides = shape.computeStrides()
        strides.contentEquals(expectedStrides)
    }

    override operator fun get(vararg indices: Int): V {
        require(indices.size == shape.dimensions.size) {
            "Number of indices (${indices.size}) must match tensor dimensions (${shape.dimensions.size})"
        }
        
        var flatIndex = offset
        for (i in indices.indices) {
            require(indices[i] >= 0 && indices[i] < shape.dimensions[i]) {
                "Index ${indices[i]} out of bounds for dimension $i with size ${shape.dimensions[i]}"
            }
            flatIndex += indices[i] * strides[i]
        }
        
        require(flatIndex >= 0 && flatIndex < parentData.size) {
            "Computed index $flatIndex is out of bounds for parent data of size ${parentData.size}"
        }
        
        return parentData[flatIndex]
    }

    override fun copyTo(dest: Array<V>, destOffset: Int) {
        var destIndex = destOffset
        iterateAll { flatIndex ->
            dest[destIndex++] = parentData[flatIndex]
        }
    }

    override fun slice(ranges: IntArray): TensorData<T, V> {
        require(ranges.size == shape.dimensions.size * 2) {
            "Ranges array must contain start,end pairs for each dimension. Expected ${shape.dimensions.size * 2}, got ${ranges.size}"
        }
        
        // Calculate new shape and strides
        val newDimensions = mutableListOf<Int>()
        val newStrides = mutableListOf<Int>()
        var newOffset = offset
        
        for (i in shape.dimensions.indices) {
            val start = ranges[i * 2]
            val end = ranges[i * 2 + 1]
            
            require(start >= 0 && start < shape.dimensions[i] && end > start && end <= shape.dimensions[i]) {
                "Invalid range [$start, $end) for dimension $i with size ${shape.dimensions[i]}"
            }
            
            newDimensions.add(end - start)
            newStrides.add(strides[i])
            newOffset += start * strides[i]
        }
        
        val newShape = Shape(newDimensions.toIntArray())
        return ViewTensorData(parentData, newShape, newStrides.toIntArray(), newOffset, parentShape)
    }

    override fun materialize(): TensorData<T, V> {
        if (isContiguous && offset == 0 && shape.volume == parentData.size) {
            // Already effectively materialized
            return DenseTensorData(shape, parentData, strides, offset)
        }
        
        // Create a new contiguous array with the view's data
        @Suppress("UNCHECKED_CAST")
        val materializedData = arrayOfNulls<Any>(shape.volume) as Array<V>
        copyTo(materializedData, 0)
        
        return DenseTensorData(shape, materializedData)
    }

    /**
     * Iterates over all elements in the view using stride-based indexing.
     */
    private fun iterateAll(action: (flatIndex: Int) -> Unit) {
        val indices = IntArray(shape.dimensions.size)
        iterateRecursive(0, offset, indices, action)
    }
    
    private fun iterateRecursive(dim: Int, currentOffset: Int, indices: IntArray, action: (flatIndex: Int) -> Unit) {
        if (dim == shape.dimensions.size) {
            action(currentOffset)
            return
        }
        
        for (i in 0 until shape.dimensions[dim]) {
            indices[dim] = i
            iterateRecursive(dim + 1, currentOffset + i * strides[dim], indices, action)
        }
    }
}