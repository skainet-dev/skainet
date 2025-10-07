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
public interface Tensor<T : DType> {
    /**
     * The data storage component of this tensor.
     * Provides access to the underlying tensor data and storage operations.
     */
    public val data: TensorData<T>

    /**
     * The operations component of this tensor.
     * Provides mathematical operations that can be performed on tensors.
     */
    public val ops: TensorOps<T>

    /**
     * Copies the tensor data to the destination array, delegated to the underlying data.
     */
    public fun <V> copyTo(dest: Array<V>, destOffset: Int = 0): Unit = data.copyTo(dest, destOffset)

    /**
     * Creates a slice view of this tensor data, delegated to the underlying data.
     */
    public fun slice(ranges: IntArray): TensorData<T> = data.slice(ranges)

    /**
     * Materializes this tensor data into a contiguous representation, delegated to the underlying data.
     */
    public fun materialize(): TensorData<T> = data.materialize()

    // Delegated TensorOps methods
    /**
     * Adds another tensor element-wise, delegated to the operations component.
     */
    public operator fun plus(other: Tensor<T>): Tensor<T> = ops.run { plus(other) }

    /**
     * Subtracts another tensor element-wise, delegated to the operations component.
     */
    public operator fun minus(other: Tensor<T>): Tensor<T> = ops.run { minus(other) }

    /**
     * Multiplies with another tensor element-wise, delegated to the operations component.
     */
    public operator fun times(other: Tensor<T>): Tensor<T> = ops.run { times(other) }

    /**
     * Divides by another tensor element-wise, delegated to the operations component.
     */
    public operator fun div(other: Tensor<T>): Tensor<T> = ops.run { div(other) }

    // Scalar operations - Int
    public operator fun plus(scalar: Int): Tensor<T> = ops.run { plus(scalar) }
    public operator fun minus(scalar: Int): Tensor<T> = ops.run { minus(scalar) }
    public operator fun times(scalar: Int): Tensor<T> = ops.run { times(scalar) }
    public operator fun div(scalar: Int): Tensor<T> = ops.run { div(scalar) }

    // Scalar operations - Float
    public operator fun plus(scalar: Float): Tensor<T> = ops.run { plus(scalar) }
    public operator fun minus(scalar: Float): Tensor<T> = ops.run { minus(scalar) }
    public operator fun times(scalar: Float): Tensor<T> = ops.run { times(scalar) }
    public operator fun div(scalar: Float): Tensor<T> = ops.run { div(scalar) }

    // Scalar operations - Double
    public operator fun plus(scalar: Double): Tensor<T> = ops.run { plus(scalar) }
    public operator fun minus(scalar: Double): Tensor<T> = ops.run { minus(scalar) }
    public operator fun times(scalar: Double): Tensor<T> = ops.run { times(scalar) }
    public operator fun div(scalar: Double): Tensor<T> = ops.run { div(scalar) }

    // Mathematical functions
    /**
     * Transpose operation, delegated to the operations component.
     */
    public fun t(): Tensor<T> = ops.run { t() }

    /**
     * ReLU activation function, delegated to the operations component.
     */
    public fun relu(): Tensor<T> = ops.run { relu() }

    /**
     * Sigmoid activation function, delegated to the operations component.
     */
    public fun sigmoid(): Tensor<T> = ops.run { sigmoid() }

    /**
     * Tanh activation function, delegated to the operations component.
     */
    public fun tanh(): Tensor<T> = ops.run { tanh() }

    /**
     * Softmax function along specified dimension, delegated to the operations component.
     */
    public fun softmax(dimension: Int): Tensor<T> = ops.run { softmax(dimension) }

    /**
     * Flatten operation, delegated to the operations component.
     */
    public fun flatten(startDim: Int = 1, endDim: Int = -1): Tensor<T> =
        ops.run { flatten(startDim, endDim) }

    /**
     * Reshape operation with Shape, delegated to the operations component.
     */
    public fun reshape(newShape: Shape): Tensor<T> = ops.run { reshape(newShape) }

    /**
     * Reshape operation with dimensions, delegated to the operations component.
     */
    public fun reshape(vararg dimensions: Int): Tensor<T> = ops.run { reshape(*dimensions) }
}

