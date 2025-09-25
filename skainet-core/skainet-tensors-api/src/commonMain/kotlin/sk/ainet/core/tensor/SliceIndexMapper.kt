package sk.ainet.core.tensor

/**
 * Index mapper implementation for basic slice operations.
 * 
 * SliceIndexMapper handles mapping from view coordinates to parent tensor coordinates
 * for sliced tensors, supporting start indices, step sizes, and dimension reductions.
 * 
 * @property parentShape The shape of the parent tensor
 * @property sliceDescriptors List of slice descriptors defining the view
 */
public class SliceIndexMapper(
    private val parentShape: Shape,
    private val sliceDescriptors: List<SliceDescriptor>
) : IndexMapper {
    
    /**
     * Maps view indices to parent tensor indices using slice descriptors.
     * 
     * @param viewIndices The indices in the view coordinate system
     * @return The corresponding indices in the parent tensor coordinate system
     */
    override fun mapToParent(viewIndices: IntArray): IntArray {
        require(viewIndices.size <= sliceDescriptors.size) {
            "View indices size ${viewIndices.size} exceeds slice descriptors size ${sliceDescriptors.size}"
        }
        
        val parentIndices = IntArray(parentShape.rank)
        var viewDimIndex = 0
        
        for (parentDimIndex in parentIndices.indices) {
            val descriptor = sliceDescriptors[parentDimIndex]
            
            when (descriptor) {
                is SliceDescriptor.Range -> {
                    require(viewDimIndex < viewIndices.size) {
                        "Not enough view indices for dimension $parentDimIndex"
                    }
                    val viewIndex = viewIndices[viewDimIndex]
                    parentIndices[parentDimIndex] = descriptor.start + viewIndex * descriptor.step
                    viewDimIndex++
                }
                is SliceDescriptor.Index -> {
                    parentIndices[parentDimIndex] = descriptor.index
                }
                is SliceDescriptor.All -> {
                    require(viewDimIndex < viewIndices.size) {
                        "Not enough view indices for dimension $parentDimIndex"
                    }
                    parentIndices[parentDimIndex] = viewIndices[viewDimIndex]
                    viewDimIndex++
                }
            }
        }
        
        // Validate parent indices are within bounds
        for (i in parentIndices.indices) {
            require(parentIndices[i] >= 0 && parentIndices[i] < parentShape[i]) {
                "Parent index ${parentIndices[i]} out of bounds for dimension $i (size: ${parentShape[i]})"
            }
        }
        
        return parentIndices
    }
}

/**
 * Sealed class representing different types of slice operations.
 */
public sealed class SliceDescriptor {
    /**
     * Represents a range slice [start:end:step].
     */
    public data class Range(
        val start: Int,
        val end: Int,
        val step: Int = 1
    ) : SliceDescriptor()
    
    /**
     * Represents a single index selection (reduces dimension).
     */
    public data class Index(val index: Int) : SliceDescriptor()
    
    /**
     * Represents selecting all elements in a dimension [:].
     */
    public object All : SliceDescriptor()
}