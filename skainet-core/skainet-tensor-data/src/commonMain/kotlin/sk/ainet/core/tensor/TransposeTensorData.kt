package sk.ainet.core.tensor

/**
 * TensorData implementation for transpose operations.
 * This provides a zero-copy transpose by reordering dimensions
 * and adjusting strides accordingly.
 */
public class TransposeTensorData<T : DType>(
    private val sourceData: TensorData<T>,
    private val permutation: IntArray
) : TensorData<T> {

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

    override operator fun <V> get(vararg indices: Int): V {
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

    override fun <V> copyTo(dest: Array<V>, destOffset: Int) {
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

    override fun slice(ranges: IntArray): TensorData<T> {
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

    public override fun <V> materialize(): TensorData<T> {
        val materializedData = arrayOfNulls<V>(shape.volume)
        copyTo(materializedData, 0)
        return this.sourceData // TensorData(materializedData.requireNoNulls(), shape)
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

    public companion object {
        /**
         * Creates a transpose that swaps the last two dimensions (common matrix transpose).
         */
        public fun <T : DType> matrixTranspose(sourceData: TensorData<T>): TransposeTensorData<T> {
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
        public fun <T : DType> fullTranspose(sourceData: TensorData<T>): TransposeTensorData<T> {
            val dims = sourceData.shape.dimensions.size
            val permutation = IntArray(dims) { dims - 1 - it }
            
            return TransposeTensorData(sourceData, permutation)
        }
    }
}