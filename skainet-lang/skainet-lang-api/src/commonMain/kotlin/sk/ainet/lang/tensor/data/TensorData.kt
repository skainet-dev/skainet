package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType

/**
 * A fundamental data structure interface that provides indexed access to elements.
 * 
 * This interface serves as the base for all data structures that need to provide
 * element access through multi-dimensional indexing. It is designed to support
 * efficient access patterns commonly used in neural network computations.
 * 
 * @param T the type of elements that can be accessed
 */
public interface ItemsAccessor<T> {
    /**
     * Retrieves an element at the specified multi-dimensional indices.
     * 
     * This operator function allows accessing elements using bracket notation
     * with variable number of indices, supporting tensors of any dimensionality.
     * 
     * @param indices the coordinates to access the element at each dimension
     * @return the element of type T at the specified indices
     * @throws IndexOutOfBoundsException if any index is out of bounds
     */
    public operator fun get(vararg indices: Int): T
}

/**
 * The fundamental data structure for tensor operations in the SKaiNET framework.
 * 
 * TensorData represents the core abstraction for all tensor-like data structures
 * used throughout the neural network computation system. It combines element access
 * capabilities with shape information, providing a unified interface for working
 * with multi-dimensional data arrays.
 * 
 * This interface serves as the foundation for:
 * - Neural network weight storage
 * - Activation value containers  
 * - Gradient computation data structures
 * - Input/output tensor representations
 * 
 * The generic type parameters allow for flexible data type support while maintaining
 * type safety across different numerical precisions and value representations.
 * 
 * @param T the data type constraint extending DType, defining the numerical precision
 * @param V the actual value type that will be stored and accessed
 */
public interface TensorData<T : DType, V> : ItemsAccessor<V> {
    /**
     * The shape descriptor that defines the dimensionality and size of this tensor data.
     * 
     * The shape property provides essential metadata about the tensor's structure,
     * including the number of dimensions and the size along each dimension. This
     * information is crucial for:
     * - Bounds checking during element access
     * - Memory layout calculations
     * - Broadcasting operations
     * - Tensor operation compatibility verification
     * 
     * @return the Shape object describing this tensor's dimensional structure
     */
    public val shape: Shape
}