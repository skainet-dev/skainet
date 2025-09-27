package sk.ainet.core.tensor

/**
 * TensorData implementation for transpose operations.
 * This provides a zero-copy transpose by reordering dimensions
 * and adjusting strides accordingly.
 */
public class TransposeTensorData<T : DType, V>(
    private val sourceData: TensorData<T, V>,
    private val permutation: IntArray
) : TensorData<T, V> {

    override val shape: Shape = computeTransposeShape()
    override val strides: IntArray = computeTransposeStrides()
    override val offset: Int = sourceData.offset
    override val isContiguous: Boolean = isTransposeContiguous()

    init {
        require(permutation.size == sourceData.shape.dimensions.size) {
            "Permutation size (${permutation.size}) must match source dimensions (${sourceData.shape.dimensions.size})"
        }
        
        require(permutation.toSet().size == permutation.size) {
            "Permutation must contain unique indices"
        }
        
        require(permutation.all { it >= 0 && it < sourceData.shape.dimensions.size }) {
            "All permutation indices must be valid dimension indices"
        }
    }

    override operator fun get(vararg indices: Int): V {
        require(indices.size == shape.dimensions.size) {
            "Number of indices (${indices.size}) must match tensor dimensions (${shape.dimensions.size})"
        }
        
        // Map transposed indices back to source indices
        val sourceIndices = IntArray(sourceData.shape.dimensions.size)
        for (i in permutation.indices) {
            sourceIndices[permutation[i]] = indices[i]
        }
        
        return sourceData.get(*sourceIndices)
    }

    override fun copyTo(dest: Array<V>, destOffset: Int) {
        if (isContiguous) {
            // Fast path for contiguous transpose
            sourceData.copyTo(dest, destOffset)
        } else {
            // Stride-based copy for non-contiguous transpose
            var destIndex = destOffset
            iterateAllIndices { indices ->
                dest[destIndex++] = this.get(*indices)
            }
        }
    }

    override fun slice(ranges: IntArray): TensorData<T, V> {
        require(ranges.size == shape.dimensions.size * 2) {
            "Ranges array must contain start,end pairs for each dimension. Expected ${shape.dimensions.size * 2}, got ${ranges.size}"
        }
        
        // Map slice ranges back to source dimensions
        val sourceRanges = IntArray(sourceData.shape.dimensions.size * 2)
        
        for (i in permutation.indices) {
            val sourceDim = permutation[i]
            val start = ranges[i * 2]
            val end = ranges[i * 2 + 1]
            
            require(start >= 0 && start < shape.dimensions[i] && end > start && end <= shape.dimensions[i]) {
                "Invalid range [$start, $end) for dimension $i with size ${shape.dimensions[i]}"
            }
            
            sourceRanges[sourceDim * 2] = start
            sourceRanges[sourceDim * 2 + 1] = end
        }
        
        val slicedSource = sourceData.slice(sourceRanges)
        
        // Create new transpose of the sliced data
        return TransposeTensorData(slicedSource, permutation)
    }

    override fun materialize(): TensorData<T, V> {
        // Create a materialized version by copying all transposed data
        @Suppress("UNCHECKED_CAST")
        val materializedData = arrayOfNulls<Any>(shape.volume) as Array<V>
        copyTo(materializedData, 0)
        
        return DenseTensorData(shape, materializedData)
    }

    /**
     * Computes the shape after transpose operation.
     */
    private fun computeTransposeShape(): Shape {
        val transposedDims = IntArray(sourceData.shape.dimensions.size)
        for (i in permutation.indices) {
            transposedDims[i] = sourceData.shape.dimensions[permutation[i]]
        }
        return Shape(transposedDims)
    }

    /**
     * Computes the strides after transpose operation.
     */
    private fun computeTransposeStrides(): IntArray {
        val transposedStrides = IntArray(sourceData.strides.size)
        for (i in permutation.indices) {
            transposedStrides[i] = sourceData.strides[permutation[i]]
        }
        return transposedStrides
    }

    /**
     * Checks if the transpose results in contiguous memory layout.
     */
    private fun isTransposeContiguous(): Boolean {
        if (!sourceData.isContiguous) return false
        
        // A transpose is contiguous if it maintains the memory ordering
        // This happens when the permutation preserves stride ordering
        val expectedStrides = shape.computeStrides()
        return strides.contentEquals(expectedStrides)
    }

    /**
     * Iterates over all indices in the transposed shape.
     */
    private fun iterateAllIndices(action: (indices: IntArray) -> Unit) {
        val totalElements = shape.volume
        for (i in 0 until totalElements) {
            val indices = flatIndexToMultiIndex(i, shape)
            action(indices)
        }
    }

    /**
     * Converts flat index to multi-dimensional indices.
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

    public companion object {
        /**
         * Creates a transpose that swaps the last two dimensions (common matrix transpose).
         */
        public fun <T : DType, V> matrixTranspose(sourceData: TensorData<T, V>): TransposeTensorData<T, V> {
            require(sourceData.shape.dimensions.size >= 2) {
                "Matrix transpose requires at least 2 dimensions"
            }
            
            val dims = sourceData.shape.dimensions.size
            val permutation = IntArray(dims) { it }
            
            // Swap last two dimensions
            val temp = permutation[dims - 1]
            permutation[dims - 1] = permutation[dims - 2]
            permutation[dims - 2] = temp
            
            return TransposeTensorData(sourceData, permutation)
        }

        /**
         * Creates a full transpose that reverses all dimensions.
         */
        public fun <T : DType, V> fullTranspose(sourceData: TensorData<T, V>): TransposeTensorData<T, V> {
            val dims = sourceData.shape.dimensions.size
            val permutation = IntArray(dims) { dims - 1 - it }
            
            return TransposeTensorData(sourceData, permutation)
        }
    }
}