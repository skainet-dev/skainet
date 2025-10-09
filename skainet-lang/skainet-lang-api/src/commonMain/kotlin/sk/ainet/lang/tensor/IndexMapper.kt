package sk.ainet.lang.tensor

/**
 * Interface for coordinate transformation between tensor view space and parent tensor space.
 * 
 * IndexMapper handles the complex task of translating multidimensional coordinates from
 * a tensor view's coordinate system to the parent tensor's coordinate system. This enables
 * efficient zero-copy slicing operations by providing a mapping layer that transforms
 * element access requests.
 * 
 * ## Coordinate Mapping Strategy
 * 
 * The mapper operates on the principle of coordinate space transformation:
 * - **Child Space**: The coordinate system as seen by the tensor view
 * - **Parent Space**: The coordinate system of the underlying parent tensor
 * - **Mapping Function**: A transformation that converts child coordinates to parent coordinates
 * 
 * ## Performance Considerations
 * 
 * IndexMappers are designed with performance in mind and provide optimization hints:
 * - Contiguity detection for vectorized operations
 * - Stride information for efficient memory access patterns
 * - Caching opportunities for repeated coordinate calculations
 * 
 * ## Implementation Guidelines
 * 
 * Implementations should:
 * - Validate coordinate bounds before transformation
 * - Cache expensive calculations when possible
 * - Provide accurate contiguity and stride information
 * - Handle edge cases gracefully (empty slices, single elements)
 * 
 * @see TensorView for the primary consumer of IndexMapper implementations
 * @see SliceIndexMapper for a general-purpose implementation
 * @see NCHWIndexMapper for NCHW layout-optimized implementation
 */
public interface IndexMapper {
    
    /**
     * Maps child (view) coordinates to parent tensor coordinates.
     * 
     * This method performs the core coordinate transformation, converting indices
     * from the view's coordinate system to the parent tensor's coordinate system.
     * The transformation must handle:
     * - Dimensional mapping (view dimensions to parent dimensions)
     * - Offset calculations (handling slice starting positions)
     * - Stride transformations (handling step sizes and memory layout)
     * 
     * ## Coordinate Validation
     * 
     * Implementations should validate that:
     * - childIndices.size matches the expected view dimensionality
     * - All child indices are within valid bounds for the view
     * - The resulting parent indices are within parent tensor bounds
     * 
     * ## Performance Requirements
     * 
     * This method is called for every element access operation, so implementations
     * must be highly optimized. Consider:
     * - Pre-computing stride multipliers
     * - Using lookup tables for common patterns
     * - Minimizing array allocations
     * 
     * @param childIndices the coordinates in the view's coordinate system
     * @return the corresponding coordinates in the parent tensor's coordinate system
     * @throws IndexOutOfBoundsException if childIndices are invalid or out of bounds
     * @throws IllegalArgumentException if childIndices has incorrect dimensionality
     */
    public fun mapToParent(childIndices: IntArray): IntArray
    
    /**
     * Determines whether the mapped region represents contiguous memory access.
     * 
     * This optimization hint indicates whether elements accessed through this
     * mapper are stored contiguously in memory within the parent tensor.
     * Contiguous access patterns enable significant performance optimizations:
     * - Vectorized operations (SIMD)
     * - Cache-friendly memory access patterns  
     * - Bulk memory operations (memcpy-style transfers)
     * - Reduced coordinate calculation overhead
     * 
     * ## Contiguity Criteria
     * 
     * A mapping is considered contiguous when:
     * - Sequential view coordinates map to sequential parent coordinates
     * - No gaps exist between mapped memory locations
     * - The memory layout follows a predictable stride pattern
     * 
     * ## Use Cases
     * 
     * Contiguous mappings are common for:
     * - Full dimension slices: `tensor[:, :, 0:10, :]`
     * - Batch extraction: `tensor[0:5, :, :, :]`
     * - Simple range slices without stepping
     * 
     * @return true if the mapping represents contiguous memory access, false otherwise
     */
    public fun isContiguous(): Boolean
    
    /**
     * Returns the stride pattern for efficient memory access.
     * 
     * Strides define the number of elements to skip in memory when moving
     * one position along each dimension. This information is crucial for:
     * - Optimizing nested loop access patterns
     * - Implementing efficient iteration algorithms  
     * - Cache-aware memory access strategies
     * - Vectorization opportunity detection
     * 
     * ## Stride Calculation
     * 
     * For a view with shape [d0, d1, d2, ...], the stride array contains:
     * - stride[0]: elements to skip when incrementing dimension 0 by 1
     * - stride[1]: elements to skip when incrementing dimension 1 by 1
     * - stride[2]: elements to skip when incrementing dimension 2 by 1
     * - ...
     * 
     * ## Memory Layout Considerations
     * 
     * Strides must account for the parent tensor's memory layout:
     * - Row-major (C-style): rightmost dimension has stride 1
     * - NCHW layout: strides typically [C×H×W, H×W, W, 1]
     * - Slicing effects: non-unit steps multiply base strides
     * 
     * @return array of stride values for each dimension, matching view dimensionality
     */
    public fun getStride(): IntArray
}