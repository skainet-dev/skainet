package sk.ainet.core.tensor

/**
 * Dense tensor data implementation for contiguous memory layout.
 * This is the most common tensor data representation where all elements
 * are stored in a contiguous array with standard stride patterns.
 */
public class DenseTensorData<T : DType, V>(
    override val shape: Shape,
    private val data: Array<V>,
    override val strides: IntArray = shape.computeStrides(),
    override val offset: Int = 0
) : TensorData<T, V> {

    override val isContiguous: Boolean = true

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
        
        return data[flatIndex]
    }

    override fun copyTo(dest: Array<V>, destOffset: Int) {
        if (isContiguous && offset == 0) {
            // Fast path for contiguous data
            data.copyInto(dest, destOffset, 0, shape.volume)
        } else {
            // Stride-based copy for non-contiguous or offset data
            var destIndex = destOffset
            iterateAll { flatIndex ->
                dest[destIndex++] = data[flatIndex]
            }
        }
    }

    override fun slice(ranges: IntArray): TensorData<T, V> {
        require(ranges.size == shape.dimensions.size * 2) {
            "Ranges array must contain start,end pairs for each dimension. Expected ${shape.dimensions.size * 2}, got ${ranges.size}"
        }
        
        // Calculate new shape and strides for the slice
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
        return ViewTensorData(data, newShape, newStrides.toIntArray(), newOffset, shape)
    }

    override fun materialize(): TensorData<T, V> {
        // Already materialized (contiguous)
        return this
    }

    /**
     * Iterates over all elements in the tensor using stride-based indexing.
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

/**
 * Computes standard row-major strides for the given shape.
 */
internal fun Shape.computeStrides(): IntArray {
    if (dimensions.isEmpty()) return intArrayOf()
    
    val strides = IntArray(dimensions.size)
    strides[dimensions.size - 1] = 1
    
    for (i in dimensions.size - 2 downTo 0) {
        strides[i] = strides[i + 1] * dimensions[i + 1]
    }
    
    return strides
}