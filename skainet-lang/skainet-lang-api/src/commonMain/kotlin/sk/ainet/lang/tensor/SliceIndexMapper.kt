package sk.ainet.lang.tensor

import sk.ainet.lang.types.DType

/**
 * General-purpose IndexMapper implementation for arbitrary slice patterns.
 * 
 * SliceIndexMapper handles coordinate transformation for any combination of slice types,
 * providing a flexible foundation for tensor slicing operations. It implements the
 * IndexMapper interface with support for NCHW memory layout and row-major storage.
 * 
 * ## Algorithm Overview
 * 
 * The mapper works by:
 * 1. **Dimension Mapping**: Maps each view dimension to parent dimension(s)
 * 2. **Offset Calculation**: Computes base offsets from slice start positions
 * 3. **Stride Application**: Applies step sizes and memory layout strides
 * 4. **Bounds Validation**: Ensures all coordinates remain within valid ranges
 * 
 * ## Memory Layout Support
 * 
 * The implementation assumes row-major memory layout with rightmost dimension
 * having stride 1. For NCHW tensors with shape [N, C, H, W], the natural
 * strides are [C×H×W, H×W, W, 1].
 * 
 * ## Performance Optimizations
 * 
 * - **Stride Pre-calculation**: Computes and caches stride patterns on construction
 * - **Contiguity Detection**: Identifies patterns that enable bulk operations
 * - **Bounds Checking**: Validates slice compatibility during construction
 * - **Efficient Indexing**: Minimizes array allocations in hot paths
 * 
 * @param T the data type constraint extending DType
 * @param V the actual value type
 * @param parentShape the shape of the parent tensor
 * @param slices the list of slice operations, one per parent dimension
 * @param viewShape the computed shape of the resulting view
 */
public class SliceIndexMapper<T : DType, V>(
    private val parentShape: Shape,
    private val slices: List<Slice<T, V>>,
    private val viewShape: Shape
) : IndexMapper {
    
    /**
     * Pre-computed strides for the parent tensor in row-major layout.
     * These strides define memory access patterns: stride[i] is the number
     * of elements to skip when moving one position along dimension i.
     */
    private val parentStrides: IntArray = computeParentStrides()
    
    /**
     * Pre-computed strides for the view tensor.
     * These strides may differ from parent strides due to slicing effects,
     * particularly when step sizes are not 1.
     */
    private val viewStrides: IntArray = computeViewStrides()
    
    /**
     * Mapping from view dimensions to parent dimensions.
     * At slices eliminate dimensions, so view dimensions may not map 1:1
     * with parent dimensions.
     */
    private val dimensionMapping: IntArray = computeDimensionMapping()
    
    /**
     * Whether this mapper represents contiguous memory access.
     * Contiguous access enables vectorized operations and bulk transfers.
     */
    private val isContiguousAccess: Boolean = computeContiguity()
    
    init {
        require(slices.size == parentShape.rank) {
            "Number of slices (${slices.size}) must match parent shape rank (${parentShape.rank})"
        }
        
        // Validate slice compatibility
        slices.forEachIndexed { dim, slice ->
            require(slice.isValid(parentShape[dim])) {
                "Invalid slice for dimension $dim (size ${parentShape[dim]}): $slice"
            }
        }
    }
    
    /**
     * Maps view coordinates to parent tensor coordinates.
     * 
     * This implementation handles all slice types:
     * - **Range**: Maps view index to parent range with offset
     * - **At**: Uses fixed parent index (view dimension eliminated)
     * - **All**: Direct mapping (view index = parent index)
     * - **Step**: Applies stride multiplication to view index
     * 
     * @param childIndices coordinates in view space
     * @return corresponding coordinates in parent space
     */
    override fun mapToParent(childIndices: IntArray): IntArray {
        require(childIndices.size == viewShape.rank) {
            "Expected ${viewShape.rank} view indices, got ${childIndices.size}"
        }
        
        val parentIndices = IntArray(parentShape.rank)
        var viewDim = 0
        
        for (parentDim in 0 until parentShape.rank) {
            val slice = slices[parentDim]
            
            parentIndices[parentDim] = when (slice) {
                is Slice.Range -> {
                    val viewIndex = childIndices[viewDim]
                    require(viewIndex >= 0 && viewIndex < (slice.end - slice.start)) {
                        "View index $viewIndex out of range for slice $slice"
                    }
                    slice.start + viewIndex
                }
                
                is Slice.At -> {
                    // At slices use fixed parent index and don't consume view dimensions
                    slice.index
                }
                
                is Slice.All -> {
                    val viewIndex = childIndices[viewDim]
                    require(viewIndex >= 0 && viewIndex < parentShape[parentDim]) {
                        "View index $viewIndex out of bounds for dimension $parentDim"
                    }
                    viewIndex
                }
                
                is Slice.Step -> {
                    val viewIndex = childIndices[viewDim]
                    val expectedSize = (slice.end - slice.start + slice.step - 1) / slice.step
                    require(viewIndex >= 0 && viewIndex < expectedSize) {
                        "View index $viewIndex out of range for step slice $slice"
                    }
                    slice.start + (viewIndex * slice.step)
                }
            }
            
            // Only increment view dimension for non-At slices
            if (slice !is Slice.At) {
                viewDim++
            }
        }
        
        return parentIndices
    }
    
    /**
     * Returns whether this mapper represents contiguous memory access.
     */
    override fun isContiguous(): Boolean = isContiguousAccess
    
    /**
     * Returns the stride pattern for the view.
     */
    override fun getStride(): IntArray = viewStrides.copyOf()
    
    /**
     * Returns whether this slice pattern can benefit from vectorized access.
     * 
     * Vectorization is beneficial when:
     * - Memory access is contiguous
     * - Access patterns follow natural tensor layout (NCHW)
     * - No complex stride patterns that would prevent SIMD operations
     */
    public fun canVectorize(): Boolean {
        return isContiguousAccess && !hasComplexStrides()
    }
    
    /**
     * Checks if the slice pattern has complex stride patterns that prevent vectorization.
     */
    private fun hasComplexStrides(): Boolean {
        return slices.any { slice ->
            when (slice) {
                is Slice.Step -> slice.step > 1
                else -> false
            }
        }
    }
    
    /**
     * Pre-calculates memory access patterns for common slice types.
     * This enables branch prediction optimization and faster hot path execution.
     * 
     * @return AccessPattern describing the memory access characteristics
     */
    public fun getAccessPattern(): AccessPattern {
        val pattern = when {
            // Fast path: contiguous access with simple patterns
            isContiguousAccess && !hasComplexStrides() -> AccessPattern.CONTIGUOUS_SIMPLE
            
            // Medium path: contiguous but with some complexity  
            isContiguousAccess -> AccessPattern.CONTIGUOUS_COMPLEX
            
            // Slow path: non-contiguous access requiring individual element fetches
            else -> AccessPattern.NON_CONTIGUOUS
        }
        
        return pattern
    }
    
    /**
     * Describes the memory access pattern for optimization decisions.
     */
    public enum class AccessPattern {
        /** Contiguous memory access with simple patterns - use vectorized operations */
        CONTIGUOUS_SIMPLE,
        
        /** Contiguous memory access but with complex patterns - use optimized scalar operations */  
        CONTIGUOUS_COMPLEX,
        
        /** Non-contiguous memory access - use individual element access */
        NON_CONTIGUOUS
    }
    
    /**
     * Computes parent tensor strides assuming row-major layout.
     * For a tensor with shape [d0, d1, d2, ...], strides are:
     * [d1×d2×..., d2×..., d3×..., ..., 1]
     */
    private fun computeParentStrides(): IntArray {
        val strides = IntArray(parentShape.rank)
        var stride = 1
        
        // Build strides from rightmost dimension
        for (i in parentShape.rank - 1 downTo 0) {
            strides[i] = stride
            stride *= parentShape[i]
        }
        
        return strides
    }
    
    /**
     * Computes view tensor strides based on slice patterns.
     * View strides may differ from parent strides due to:
     * - Step slices multiplying base strides
     * - Dimension elimination from At slices
     * - Range slices preserving natural strides
     */
    private fun computeViewStrides(): IntArray {
        val strides = IntArray(viewShape.rank)
        var viewDim = 0
        
        for (parentDim in 0 until parentShape.rank) {
            val slice = slices[parentDim]
            
            if (slice !is Slice.At) {
                val baseStride = parentStrides[parentDim]
                strides[viewDim] = when (slice) {
                    is Slice.Step -> baseStride * slice.step
                    else -> baseStride
                }
                viewDim++
            }
        }
        
        return strides
    }
    
    /**
     * Computes mapping from view dimensions to parent dimensions.
     * This handles dimension elimination caused by At slices.
     */
    private fun computeDimensionMapping(): IntArray {
        val mapping = IntArray(viewShape.rank)
        var viewDim = 0
        
        for (parentDim in 0 until parentShape.rank) {
            if (slices[parentDim] !is Slice.At) {
                mapping[viewDim] = parentDim
                viewDim++
            }
        }
        
        return mapping
    }
    
    /**
     * Determines if the slicing pattern results in contiguous memory access.
     * 
     * Contiguity is preserved when:
     * - No step slices with step > 1
     * - Range and All slices maintain natural order
     * - At slices don't break contiguity of remaining dimensions
     */
    private fun computeContiguity(): Boolean {
        // Check for non-unit steps
        if (slices.any { it.hasNonTrivialStride() }) {
            return false
        }
        
        // Check for gaps in dimensional access
        // This is a simplified check; more sophisticated analysis could
        // detect additional contiguous patterns
        var lastContiguousDim = -1
        
        for (parentDim in parentShape.rank - 1 downTo 0) {
            val slice = slices[parentDim]
            
            when (slice) {
                is Slice.Range -> {
                    if (slice.start == 0 && slice.end == parentShape[parentDim]) {
                        // Full dimension access
                        lastContiguousDim = parentDim
                    } else if (lastContiguousDim == -1) {
                        // Partial access on rightmost accessed dimension is OK
                        lastContiguousDim = parentDim
                    } else {
                        // Partial access with more dimensions to the right breaks contiguity
                        return false
                    }
                }
                
                is Slice.All -> {
                    // Full dimension access preserves contiguity
                    lastContiguousDim = parentDim
                }
                
                is Slice.At -> {
                    // At slices can break contiguity if not on leftmost dimensions
                    if (lastContiguousDim != -1) {
                        return false
                    }
                }
                
                is Slice.Step -> {
                    // Non-unit steps already handled above
                    return false
                }
            }
        }
        
        return true
    }
}