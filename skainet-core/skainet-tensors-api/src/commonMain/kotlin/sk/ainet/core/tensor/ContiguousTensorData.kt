package sk.ainet.core.tensor

/**
 * Optimized TensorData implementation for contiguous memory layout.
 * This implementation provides fast-path optimizations for operations
 * on densely packed tensor data.
 */
public class ContiguousTensorData<T : DType, V>(
    override val shape: Shape,
    private val data: Array<V>,
    override val offset: Int = 0
) : TensorData<T, V> {

    override val strides: IntArray = shape.computeStrides()
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
        // Fast path for contiguous data - direct array copy
        if (offset == 0 && shape.volume == data.size) {
            data.copyInto(dest, destOffset, 0, shape.volume)
        } else {
            // Copy with offset consideration
            data.copyInto(dest, destOffset, offset, offset + shape.volume)
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
        
        // Check if the result would still be contiguous
        val expectedStrides = newShape.computeStrides()
        if (newStrides.toIntArray().contentEquals(expectedStrides)) {
            return ContiguousTensorData(newShape, data, newOffset)
        } else {
            return ViewTensorData(data, newShape, newStrides.toIntArray(), newOffset, shape)
        }
    }

    override fun materialize(): TensorData<T, V> {
        // Already materialized (contiguous)
        return this
    }
}

/**
 * Computes standard row-major strides for the given shape.
 */
private fun Shape.computeStrides(): IntArray {
    if (dimensions.isEmpty()) return intArrayOf()
    
    val strides = IntArray(dimensions.size)
    strides[dimensions.size - 1] = 1
    
    for (i in dimensions.size - 2 downTo 0) {
        strides[i] = strides[i + 1] * dimensions[i + 1]
    }
    
    return strides
}