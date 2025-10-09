package sk.ainet.lang.tensor

import sk.ainet.lang.types.DType

/**
 * Specialized IndexMapper optimized for NCHW (Batch, Channel, Height, Width) tensor layout.
 * 
 * NCHWIndexMapper provides highly optimized coordinate transformation for 4-dimensional tensors
 * following the NCHW memory layout convention commonly used in deep learning frameworks.
 * This layout stores data in row-major order with the fastest-changing dimension being Width.
 * 
 * ## NCHW Memory Layout
 * 
 * For a tensor with shape [N, C, H, W], elements are stored with strides:
 * - **Batch (N)**: stride = C × H × W (largest stride)
 * - **Channel (C)**: stride = H × W  
 * - **Height (H)**: stride = W
 * - **Width (W)**: stride = 1 (fastest changing)
 * 
 * ## Optimization Characteristics
 * 
 * This mapper is specifically optimized for common NCHW access patterns:
 * - **Batch slicing**: `tensor[0:k, :, :, :]` - Highly efficient, large contiguous blocks
 * - **Channel extraction**: `tensor[:, 0:k, :, :]` - Very efficient, exploits NCHW layout
 * - **Spatial regions**: `tensor[:, :, h1:h2, w1:w2]` - Good efficiency, localized access  
 * - **Width slicing**: `tensor[:, :, :, w1:w2]` - Moderate efficiency, may require striding
 * 
 * ## Performance Benefits
 * 
 * - **Pre-calculated Strides**: Eliminates runtime stride computation
 * - **Specialized Algorithms**: Optimized coordinate mapping for each dimension
 * - **Cache-Friendly Access**: Takes advantage of NCHW spatial locality
 * - **Vectorization Hints**: Provides accurate contiguity information
 * 
 * @param T the data type constraint extending DType
 * @param V the actual value type
 * @param parentShape the NCHW shape of the parent tensor [N, C, H, W]
 * @param slices the list of slice operations for each NCHW dimension
 * @param viewShape the computed shape of the resulting view
 * @throws IllegalArgumentException if parentShape is not 4-dimensional
 */
public class NCHWIndexMapper<T : DType, V>(
    private val parentShape: Shape,
    private val slices: List<Slice<T, V>>,
    private val viewShape: Shape
) : IndexMapper {
    
    // NCHW dimension indices for clarity
    private val N_DIM = 0 // Batch dimension
    private val C_DIM = 1 // Channel dimension  
    private val H_DIM = 2 // Height dimension
    private val W_DIM = 3 // Width dimension
    
    /**
     * Pre-calculated strides for NCHW layout: [C×H×W, H×W, W, 1]
     */
    private val nchwStrides: IntArray
    
    /**
     * View strides accounting for slice effects
     */
    private val viewStrides: IntArray
    
    /**
     * Dimension mapping from view space to parent space
     */
    private val dimensionMapping: IntArray
    
    /**
     * Contiguity analysis result
     */
    private val isContiguousAccess: Boolean
    
    init {
        require(parentShape.rank == 4) {
            "NCHWIndexMapper requires 4-dimensional parent tensor (NCHW), got rank ${parentShape.rank}"
        }
        
        require(slices.size == 4) {
            "NCHWIndexMapper requires exactly 4 slices for NCHW dimensions, got ${slices.size}"
        }
        
        // Validate slices against NCHW dimensions
        slices.forEachIndexed { dim, slice ->
            require(slice.isValid(parentShape[dim])) {
                "Invalid slice for NCHW dimension $dim (size ${parentShape[dim]}): $slice"
            }
        }
        
        // Pre-calculate all mapping information
        nchwStrides = computeNCHWStrides()
        viewStrides = computeViewStrides()
        dimensionMapping = computeDimensionMapping()
        isContiguousAccess = computeContiguity()
    }
    
    /**
     * Highly optimized coordinate mapping for NCHW layout.
     * 
     * This implementation leverages knowledge of NCHW dimension semantics
     * to provide the fastest possible coordinate transformation. Each
     * dimension is handled with specialized logic.
     */
    override fun mapToParent(childIndices: IntArray): IntArray {
        require(childIndices.size == viewShape.rank) {
            "Expected ${viewShape.rank} view indices, got ${childIndices.size}"
        }
        
        val parentIndices = IntArray(4) // NCHW always has 4 dimensions
        var viewDim = 0
        
        // Process each NCHW dimension with specialized logic
        for (nchwDim in 0 until 4) {
            val slice = slices[nchwDim]
            
            parentIndices[nchwDim] = when (slice) {
                is Slice.Range -> {
                    // Range slicing: common for batch/channel extraction
                    val viewIndex = childIndices[viewDim]
                    validateRangeIndex(viewIndex, slice, nchwDim)
                    slice.start + viewIndex
                }
                
                is Slice.At -> {
                    // At slicing: common for single batch/channel selection
                    slice.index
                }
                
                is Slice.All -> {
                    // All slicing: preserves dimension, direct mapping
                    val viewIndex = childIndices[viewDim]
                    validateAllIndex(viewIndex, nchwDim)
                    viewIndex
                }
                
                is Slice.Step -> {
                    // Step slicing: less common but supported for subsampling
                    val viewIndex = childIndices[viewDim]
                    validateStepIndex(viewIndex, slice, nchwDim)
                    slice.start + (viewIndex * slice.step)
                }
            }
            
            // Advance view dimension for non-At slices
            if (slice !is Slice.At) {
                viewDim++
            }
        }
        
        return parentIndices
    }
    
    /**
     * Returns contiguity information optimized for NCHW patterns.
     */
    override fun isContiguous(): Boolean = isContiguousAccess
    
    /**
     * Returns NCHW-optimized view strides.
     */
    override fun getStride(): IntArray = viewStrides.copyOf()
    
    /**
     * Computes the natural NCHW strides: [C×H×W, H×W, W, 1]
     */
    private fun computeNCHWStrides(): IntArray {
        val N = parentShape[N_DIM]  // Batch size
        val C = parentShape[C_DIM]  // Channel count
        val H = parentShape[H_DIM]  // Height
        val W = parentShape[W_DIM]  // Width
        
        return intArrayOf(
            C * H * W,  // Batch stride: skip entire [C,H,W] blocks
            H * W,      // Channel stride: skip [H,W] feature maps  
            W,          // Height stride: skip W elements per row
            1           // Width stride: adjacent elements
        )
    }
    
    /**
     * Computes view strides accounting for NCHW slice effects.
     */
    private fun computeViewStrides(): IntArray {
        val strides = mutableListOf<Int>()
        
        for (nchwDim in 0 until 4) {
            val slice = slices[nchwDim]
            
            if (slice !is Slice.At) {
                val baseStride = nchwStrides[nchwDim]
                val effectiveStride = when (slice) {
                    is Slice.Step -> baseStride * slice.step
                    else -> baseStride
                }
                strides.add(effectiveStride)
            }
        }
        
        return strides.toIntArray()
    }
    
    /**
     * Computes dimension mapping for NCHW layout.
     */
    private fun computeDimensionMapping(): IntArray {
        val mapping = mutableListOf<Int>()
        
        for (nchwDim in 0 until 4) {
            if (slices[nchwDim] !is Slice.At) {
                mapping.add(nchwDim)
            }
        }
        
        return mapping.toIntArray()
    }
    
    /**
     * Analyzes contiguity for NCHW-specific patterns.
     * 
     * NCHW layout has specific contiguity characteristics:
     * - Batch slicing preserves contiguity (large blocks)
     * - Channel slicing is highly efficient  
     * - Spatial slicing may be contiguous for full-width operations
     * - Width-only slicing may introduce striding
     */
    private fun computeContiguity(): Boolean {
        // Check for step slices that break contiguity
        if (slices.any { it.hasNonTrivialStride() }) {
            return false
        }
        
        // Analyze NCHW-specific contiguity patterns
        val batchSlice = slices[N_DIM]
        val channelSlice = slices[C_DIM]  
        val heightSlice = slices[H_DIM]
        val widthSlice = slices[W_DIM]
        
        // Contiguous patterns in NCHW:
        // 1. Full or range access to rightmost dimensions
        // 2. At slices on leftmost dimensions don't break contiguity
        
        var hasPartialAccess = false
        
        // Check from right to left (W -> H -> C -> N)
        for (dim in intArrayOf(W_DIM, H_DIM, C_DIM, N_DIM)) {
            val slice = slices[dim]
            
            when (slice) {
                is Slice.Range -> {
                    if (slice.start == 0 && slice.end == parentShape[dim]) {
                        // Full dimension access - good for contiguity
                    } else {
                        // Partial access
                        if (hasPartialAccess) {
                            // Multiple partial accesses break contiguity  
                            return false
                        }
                        hasPartialAccess = true
                    }
                }
                
                is Slice.All -> {
                    // Full access preserves contiguity
                }
                
                is Slice.At -> {
                    // At slices are OK if no partial access encountered yet
                    if (hasPartialAccess) {
                        return false
                    }
                }
                
                is Slice.Step -> {
                    // Steps break contiguity (already checked above)
                    return false
                }
            }
        }
        
        return true
    }
    
    // Validation helpers with NCHW-specific error messages
    
    private fun validateRangeIndex(viewIndex: Int, slice: Slice.Range<T, V>, nchwDim: Int) {
        val dimName = getDimensionName(nchwDim)
        val rangeSize = slice.end - slice.start
        require(viewIndex >= 0 && viewIndex < rangeSize) {
            "View index $viewIndex out of range for $dimName range slice $slice"
        }
    }
    
    private fun validateAllIndex(viewIndex: Int, nchwDim: Int) {
        val dimName = getDimensionName(nchwDim)
        val dimSize = parentShape[nchwDim]
        require(viewIndex >= 0 && viewIndex < dimSize) {
            "View index $viewIndex out of bounds for $dimName dimension (size $dimSize)"
        }
    }
    
    private fun validateStepIndex(viewIndex: Int, slice: Slice.Step<T, V>, nchwDim: Int) {
        val dimName = getDimensionName(nchwDim)
        val expectedSize = (slice.end - slice.start + slice.step - 1) / slice.step
        require(viewIndex >= 0 && viewIndex < expectedSize) {
            "View index $viewIndex out of range for $dimName step slice $slice"
        }
    }
    
    private fun getDimensionName(nchwDim: Int): String = when (nchwDim) {
        N_DIM -> "Batch(N)"
        C_DIM -> "Channel(C)"  
        H_DIM -> "Height(H)"
        W_DIM -> "Width(W)"
        else -> "Unknown($nchwDim)"
    }
}