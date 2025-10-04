package sk.ainet.core.tensor

/**
 * Interface representing a multi-dimensional array of numeric values.
 *
 * A Tensor is a generalization of vectors and matrices to potentially higher dimensions.
 * The [shape] property defines the dimensions of the tensor, and the [get] operator
 * allows accessing individual elements by their indices.
 *
 * This interface uses composition instead of inheritance to separate data storage
 * and mathematical operations, allowing for better modularity and testability.
 */
public interface Tensor<T : DType, V> {
    /**
     * The data storage component of this tensor.
     * Provides access to the underlying tensor data and storage operations.
     */
    public val data: TensorData<T, V>
    
    /**
     * The operations component of this tensor.
     * Provides mathematical operations that can be performed on tensors.
     */
    public val ops: TensorOps<T, V, Tensor<T, V>>
    
    // Delegated TensorData properties and methods
    /**
     * The shape of the tensor, delegated to the underlying data.
     */
    public val shape: Shape get() = data.shape
    
    /**
     * Stride information for each dimension, delegated to the underlying data.
     */
    public val strides: IntArray get() = data.strides
    
    /**
     * Offset in the underlying data array, delegated to the underlying data.
     */
    public val offset: Int get() = data.offset
    
    /**
     * Whether this tensor data represents contiguous data in memory, delegated to the underlying data.
     */
    public val isContiguous: Boolean get() = data.isContiguous
    
    /**
     * Retrieves the value at the specified indices, delegated to the underlying data.
     */
    public operator fun get(vararg indices: Int): V = data.get(*indices)
    
    /**
     * Copies the tensor data to the destination array, delegated to the underlying data.
     */
    public fun copyTo(dest: Array<V>, destOffset: Int = 0): Unit = data.copyTo(dest, destOffset)
    
    /**
     * Creates a slice view of this tensor data, delegated to the underlying data.
     */
    public fun slice(ranges: IntArray): TensorData<T, V> = data.slice(ranges)
    
    /**
     * Materializes this tensor data into a contiguous representation, delegated to the underlying data.
     */
    public fun materialize(): TensorData<T, V> = data.materialize()
    
    // Delegated TensorOps methods
    /**
     * Adds another tensor element-wise, delegated to the operations component.
     */
    public operator fun plus(other: Tensor<T, V>): Tensor<T, V> = ops.run { this@Tensor.plus(other) }
    
    /**
     * Subtracts another tensor element-wise, delegated to the operations component.
     */
    public operator fun minus(other: Tensor<T, V>): Tensor<T, V> = ops.run { this@Tensor.minus(other) }
    
    /**
     * Multiplies with another tensor element-wise, delegated to the operations component.
     */
    public operator fun times(other: Tensor<T, V>): Tensor<T, V> = ops.run { this@Tensor.times(other) }
    
    /**
     * Divides by another tensor element-wise, delegated to the operations component.
     */
    public operator fun div(other: Tensor<T, V>): Tensor<T, V> = ops.run { this@Tensor.div(other) }
    
    // Scalar operations - Int
    public operator fun plus(scalar: Int): Tensor<T, V> = ops.run { this@Tensor.plus(scalar) }
    public operator fun minus(scalar: Int): Tensor<T, V> = ops.run { this@Tensor.minus(scalar) }
    public operator fun times(scalar: Int): Tensor<T, V> = ops.run { this@Tensor.times(scalar) }
    public operator fun div(scalar: Int): Tensor<T, V> = ops.run { this@Tensor.div(scalar) }
    
    // Scalar operations - Float
    public operator fun plus(scalar: Float): Tensor<T, V> = ops.run { this@Tensor.plus(scalar) }
    public operator fun minus(scalar: Float): Tensor<T, V> = ops.run { this@Tensor.minus(scalar) }
    public operator fun times(scalar: Float): Tensor<T, V> = ops.run { this@Tensor.times(scalar) }
    public operator fun div(scalar: Float): Tensor<T, V> = ops.run { this@Tensor.div(scalar) }
    
    // Scalar operations - Double
    public operator fun plus(scalar: Double): Tensor<T, V> = ops.run { this@Tensor.plus(scalar) }
    public operator fun minus(scalar: Double): Tensor<T, V> = ops.run { this@Tensor.minus(scalar) }
    public operator fun times(scalar: Double): Tensor<T, V> = ops.run { this@Tensor.times(scalar) }
    public operator fun div(scalar: Double): Tensor<T, V> = ops.run { this@Tensor.div(scalar) }
    
    // Mathematical functions
    /**
     * Transpose operation, delegated to the operations component.
     */
    public fun t(): Tensor<T, V> = ops.run { this@Tensor.t() }
    
    /**
     * ReLU activation function, delegated to the operations component.
     */
    public fun relu(): Tensor<T, V> = ops.run { this@Tensor.relu() }
    
    /**
     * Sigmoid activation function, delegated to the operations component.
     */
    public fun sigmoid(): Tensor<T, V> = ops.run { this@Tensor.sigmoid() }
    
    /**
     * Tanh activation function, delegated to the operations component.
     */
    public fun tanh(): Tensor<T, V> = ops.run { this@Tensor.tanh() }
    
    /**
     * Softmax function along specified dimension, delegated to the operations component.
     */
    public fun softmax(dimension: Int): Tensor<T, V> = ops.run { this@Tensor.softmax(dimension) }
    
    /**
     * Flatten operation, delegated to the operations component.
     */
    public fun flatten(startDim: Int = 1, endDim: Int = -1): Tensor<T, V> = ops.run { this@Tensor.flatten(startDim, endDim) }
    
    /**
     * Reshape operation with Shape, delegated to the operations component.
     */
    public fun reshape(newShape: Shape): Tensor<T, V> = ops.run { this@Tensor.reshape(newShape) }
    
    /**
     * Reshape operation with dimensions, delegated to the operations component.
     */
    public fun reshape(vararg dimensions: Int): Tensor<T, V> = ops.run { this@Tensor.reshape(*dimensions) }
}

