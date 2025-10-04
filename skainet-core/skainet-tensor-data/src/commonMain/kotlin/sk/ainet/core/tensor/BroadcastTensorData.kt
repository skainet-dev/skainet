package sk.ainet.core.tensor

/**
 * TensorData implementation for broadcasting operations.
 * This allows a smaller tensor to be logically expanded to match
 * a larger shape without actually copying the data.
 */
public class BroadcastTensorData<T : DType, V>(
    private val sourceData: TensorData<T, V>,
    override val shape: Shape
) : TensorData<T, V> {

    override val strides: IntArray = computeBroadcastStrides()
    override val offset: Int = sourceData.offset
    override val isContiguous: Boolean = false // Broadcasting is never contiguous

    init {
        require(canBroadcast(sourceData.shape, shape)) {
            "Cannot broadcast shape ${sourceData.shape} to ${shape}"
        }
    }

    override operator fun get(vararg indices: Int): V {
        require(indices.size == shape.dimensions.size) {
            "Number of indices (${indices.size}) must match tensor dimensions (${shape.dimensions.size})"
        }
        
        // Map broadcast indices to source indices
        val sourceIndices = mapBroadcastIndices(indices)
        return sourceData.get(*sourceIndices)
    }

    override fun copyTo(dest: Array<V>, destOffset: Int) {
        var destIndex = destOffset
        iterateAllIndices { indices ->
            dest[destIndex++] = this.get(*indices)
        }
    }

    override fun slice(ranges: IntArray): TensorData<T, V> {
        require(ranges.size == shape.dimensions.size * 2) {
            "Ranges array must contain start,end pairs for each dimension. Expected ${shape.dimensions.size * 2}, got ${ranges.size}"
        }
        
        // Create new broadcast shape from slice
        val newDimensions = mutableListOf<Int>()
        for (i in shape.dimensions.indices) {
            val start = ranges[i * 2]
            val end = ranges[i * 2 + 1]
            
            require(start >= 0 && start < shape.dimensions[i] && end > start && end <= shape.dimensions[i]) {
                "Invalid range [$start, $end) for dimension $i with size ${shape.dimensions[i]}"
            }
            
            newDimensions.add(end - start)
        }
        
        val newShape = Shape(newDimensions.toIntArray())
        
        // Map the slice back to source data if possible
        val sourceDimOffset = shape.dimensions.size - sourceData.shape.dimensions.size
        if (sourceDimOffset >= 0) {
            // Calculate source ranges for non-broadcast dimensions
            val sourceRanges = mutableListOf<Int>()
            
            for (i in sourceData.shape.dimensions.indices) {
                val broadcastDimIndex = sourceDimOffset + i
                if (broadcastDimIndex < ranges.size / 2) {
                    val start = ranges[broadcastDimIndex * 2]
                    val end = ranges[broadcastDimIndex * 2 + 1]
                    
                    if (sourceData.shape.dimensions[i] == 1) {
                        // Broadcast dimension - always use full range
                        sourceRanges.add(0)
                        sourceRanges.add(1)
                    } else {
                        // Non-broadcast dimension - use actual slice
                        sourceRanges.add(start)
                        sourceRanges.add(end)
                    }
                }
            }
            
            if (sourceRanges.size == sourceData.shape.dimensions.size * 2) {
                val slicedSource = sourceData.slice(sourceRanges.toIntArray())
                return BroadcastTensorData(slicedSource, newShape)
            }
        }
        
        // Fallback: materialize the broadcast data first
        @Suppress("UNCHECKED_CAST")
        val materializedData = arrayOfNulls<Any>(shape.volume) as Array<V>
        copyTo(materializedData, 0)
        
        return ViewTensorData(
            parentData = materializedData,
            shape = newShape,
            strides = newShape.computeStrides(),
            offset = 0,
            parentShape = shape
        )
    }

    override fun materialize(): TensorData<T, V> {
        // Create a materialized version by copying all broadcast data
        @Suppress("UNCHECKED_CAST")
        val materializedData = arrayOfNulls<Any>(shape.volume) as Array<V>
        copyTo(materializedData, 0)
        
        return DenseTensorData(shape, materializedData)
    }

    /**
     * Computes strides for broadcasting operation.
     */
    private fun computeBroadcastStrides(): IntArray {
        val resultStrides = IntArray(shape.dimensions.size)
        val sourceStrides = sourceData.strides
        
        // Align dimensions from the right (trailing dimensions)
        val sourceDimOffset = shape.dimensions.size - sourceData.shape.dimensions.size
        
        for (i in shape.dimensions.indices) {
            val sourceIndex = i - sourceDimOffset
            
            if (sourceIndex >= 0 && sourceIndex < sourceData.shape.dimensions.size) {
                // Dimension exists in source
                if (sourceData.shape.dimensions[sourceIndex] == 1 && shape.dimensions[i] > 1) {
                    // Broadcasting dimension - stride is 0
                    resultStrides[i] = 0
                } else {
                    // Non-broadcasting dimension
                    resultStrides[i] = sourceStrides[sourceIndex]
                }
            } else {
                // New dimension added by broadcasting - stride is 0
                resultStrides[i] = 0
            }
        }
        
        return resultStrides
    }

    /**
     * Maps broadcast tensor indices to source tensor indices.
     */
    private fun mapBroadcastIndices(broadcastIndices: IntArray): IntArray {
        val sourceIndices = IntArray(sourceData.shape.dimensions.size)
        val sourceDimOffset = shape.dimensions.size - sourceData.shape.dimensions.size
        
        for (i in sourceData.shape.dimensions.indices) {
            val broadcastIndex = sourceDimOffset + i
            val sourceIndex = if (broadcastIndex >= 0 && broadcastIndex < broadcastIndices.size) {
                if (sourceData.shape.dimensions[i] == 1) {
                    // Broadcasting dimension - always use index 0
                    0
                } else {
                    // Non-broadcasting dimension
                    broadcastIndices[broadcastIndex]
                }
            } else {
                0
            }
            sourceIndices[i] = sourceIndex
        }
        
        return sourceIndices
    }

    /**
     * Iterates over all indices in the broadcast shape.
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
     * Checks if sourceShape can be broadcast to targetShape.
     */
    private fun canBroadcast(sourceShape: Shape, targetShape: Shape): Boolean {
        val sourceDims = sourceShape.dimensions
        val targetDims = targetShape.dimensions
        
        if (sourceDims.size > targetDims.size) return false
        
        val offset = targetDims.size - sourceDims.size
        
        for (i in sourceDims.indices) {
            val sourceSize = sourceDims[i]
            val targetSize = targetDims[offset + i]
            
            if (sourceSize != 1 && sourceSize != targetSize) {
                return false
            }
        }
        
        return true
    }
}