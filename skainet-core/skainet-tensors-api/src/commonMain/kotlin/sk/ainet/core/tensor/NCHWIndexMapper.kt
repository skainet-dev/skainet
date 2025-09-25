package sk.ainet.core.tensor

/**
 * Index mapper optimized for NCHW (batch, channels, height, width) memory layout.
 * 
 * NCHWIndexMapper provides efficient index mapping for tensors stored in row-major
 * NCHW format, which is commonly used in deep learning frameworks for image data.
 * This mapper optimizes for the memory access patterns typical in CNN operations.
 * 
 * @property parentShape The shape of the parent tensor (must be 4D: [N, C, H, W])
 * @property viewOffset The starting offset in each dimension [n_offset, c_offset, h_offset, w_offset]
 * @property viewStrides The stride in each dimension [n_stride, c_stride, h_stride, w_stride]
 */
public class NCHWIndexMapper(
    private val parentShape: Shape,
    private val viewOffset: IntArray,
    private val viewStrides: IntArray
) : IndexMapper {
    
    init {
        require(parentShape.rank == 4) { 
            "NCHW format requires 4D tensors, got ${parentShape.rank}D" 
        }
        require(viewOffset.size == 4) { 
            "View offset must have 4 elements for NCHW, got ${viewOffset.size}" 
        }
        require(viewStrides.size == 4) { 
            "View strides must have 4 elements for NCHW, got ${viewStrides.size}" 
        }
    }
    
    // Pre-computed strides for row-major NCHW layout
    private val parentStrides: IntArray = computeRowMajorStrides(parentShape)
    
    /**
     * Maps view indices to parent tensor indices using NCHW-optimized calculations.
     * 
     * @param viewIndices The indices in the view coordinate system [n, c, h, w]
     * @return The corresponding indices in the parent tensor coordinate system
     */
    override fun mapToParent(viewIndices: IntArray): IntArray {
        require(viewIndices.size == 4) {
            "NCHW view requires 4 indices [n, c, h, w], got ${viewIndices.size}"
        }
        
        val parentIndices = IntArray(4)
        
        // Map each dimension: parent_index = view_offset + view_index * view_stride
        for (i in 0..3) {
            parentIndices[i] = viewOffset[i] + viewIndices[i] * viewStrides[i]
        }
        
        // Bounds checking for NCHW dimensions
        validateNCHWIndices(parentIndices, parentShape)
        
        return parentIndices
    }
    
    /**
     * Computes the linear memory offset for given NCHW indices.
     * This is useful for direct memory access optimizations.
     * 
     * @param indices The NCHW indices [n, c, h, w]
     * @return The linear memory offset
     */
    public fun computeLinearOffset(indices: IntArray): Int {
        require(indices.size == 4) { "NCHW requires 4 indices" }
        
        val n = indices[0]
        val c = indices[1] 
        val h = indices[2]
        val w = indices[3]
        
        return n * parentStrides[0] + c * parentStrides[1] + h * parentStrides[2] + w * parentStrides[3]
    }
    
    /**
     * Checks if the view represents contiguous memory access.
     * Contiguous access occurs when view strides match parent strides.
     */
    public fun isContiguousAccess(): Boolean {
        return viewStrides.contentEquals(intArrayOf(1, 1, 1, 1))
    }
    
    /**
     * Optimizes channel extraction by checking if only the channel dimension is sliced.
     */
    public fun isChannelSliceOnly(): Boolean {
        return viewStrides[0] == 1 && viewStrides[2] == 1 && viewStrides[3] == 1 &&
               viewOffset[0] == 0 && viewOffset[2] == 0 && viewOffset[3] == 0
    }
    
    private fun computeRowMajorStrides(shape: Shape): IntArray {
        val strides = IntArray(4)
        strides[3] = 1  // W stride
        strides[2] = shape[3]  // H stride = W
        strides[1] = shape[2] * shape[3]  // C stride = H * W
        strides[0] = shape[1] * shape[2] * shape[3]  // N stride = C * H * W
        return strides
    }
    
    private fun validateNCHWIndices(indices: IntArray, shape: Shape) {
        val dimensions = arrayOf("batch", "channel", "height", "width")
        for (i in indices.indices) {
            require(indices[i] >= 0 && indices[i] < shape[i]) {
                "Index ${indices[i]} out of bounds for ${dimensions[i]} dimension (size: ${shape[i]})"
            }
        }
    }
}

/**
 * Helper functions for creating NCHW-optimized views
 */
public object NCHWViewHelper {
    
    /**
     * Creates an index mapper for batch slicing: tensor[n1:n2, :, :, :]
     */
    public fun createBatchSlice(
        parentShape: Shape, 
        startBatch: Int, 
        endBatch: Int, 
        step: Int = 1
    ): NCHWIndexMapper {
        return NCHWIndexMapper(
            parentShape = parentShape,
            viewOffset = intArrayOf(startBatch, 0, 0, 0),
            viewStrides = intArrayOf(step, 1, 1, 1)
        )
    }
    
    /**
     * Creates an index mapper for channel extraction: tensor[:, c, :, :]
     */
    public fun createChannelExtraction(
        parentShape: Shape, 
        channelIndex: Int
    ): NCHWIndexMapper {
        return NCHWIndexMapper(
            parentShape = parentShape,
            viewOffset = intArrayOf(0, channelIndex, 0, 0),
            viewStrides = intArrayOf(1, 0, 1, 1)  // 0 stride means dimension is collapsed
        )
    }
    
    /**
     * Creates an index mapper for spatial region slicing: tensor[:, :, h1:h2, w1:w2]
     */
    public fun createSpatialSlice(
        parentShape: Shape,
        heightStart: Int, heightEnd: Int, heightStep: Int = 1,
        widthStart: Int, widthEnd: Int, widthStep: Int = 1
    ): NCHWIndexMapper {
        return NCHWIndexMapper(
            parentShape = parentShape,
            viewOffset = intArrayOf(0, 0, heightStart, widthStart),
            viewStrides = intArrayOf(1, 1, heightStep, widthStep)
        )
    }
}