package sk.ainet.core.tensor

/**
 * Interface for mapping view indices to parent tensor indices.
 * 
 * IndexMapper provides the core functionality for converting coordinates
 * from a tensor view's coordinate system to the parent tensor's coordinate system.
 */
public interface IndexMapper {
    
    /**
     * Maps view indices to parent tensor indices.
     * 
     * @param viewIndices The indices in the view coordinate system
     * @return The corresponding indices in the parent tensor coordinate system
     * @throws IndexOutOfBoundsException if the view indices are invalid
     */
    public fun mapToParent(viewIndices: IntArray): IntArray
    
    /**
     * Validates that the given view indices are within bounds.
     * 
     * @param viewIndices The indices to validate
     * @param viewShape The shape of the view
     * @return true if indices are valid, false otherwise
     */
    public fun validateIndices(viewIndices: IntArray, viewShape: Shape): Boolean {
        if (viewIndices.size != viewShape.rank) return false
        return viewIndices.zip(viewShape.dimensions).all { (index, dimension) ->
            index >= 0 && index < dimension
        }
    }
}