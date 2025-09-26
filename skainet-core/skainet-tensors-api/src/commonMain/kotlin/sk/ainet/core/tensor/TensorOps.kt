package sk.ainet.core.tensor

/**
 * Interface representing mathematical operations on tensors.
 *
 * Keeps math separate from storage, so the same math API can work
 * with multiple backends (dense, sparse, GPU, etc.).
 */
public interface TensorOps<D : DType, V, T : Tensor<D, V>> {
    /**
     * Performs matrix multiplication of two tensors.
     *
     * @param a The first tensor.
     * @param b The second tensor.
     * @return The result of matrix multiplication.
     */
    public fun matmul(a: T, b: T): T

    /**
     * Performs 4D tensor multiplication with batch processing.
     * Supports various patterns:
     * - Batch-wise: (B,M,K) × (B,K,N) → (B,M,N)
     * - Spatial: (B,C,H,W) × (B,C,W,K) → (B,C,H,K)
     * - Channel-wise: (B,C1,H,W) × (C1,C2) → (B,C2,H,W) where second tensor is reshaped internally
     *
     * @param a The first 4D tensor.
     * @param b The second tensor (4D or 2D).
     * @return The result of 4D matrix multiplication.
     */
    public fun matmul4d(a: T, b: T): T

    /**
     * Scales a tensor by a scalar value.
     *
     * @param a The tensor to scale.
     * @param scalar The scalar value to scale by.
     * @return The scaled tensor.
     */
    public fun scale(a: T, scalar: Double): T

    /**
     * Computes the dot product of two tensors.
     *
     * @param a The first tensor.
     * @param b The second tensor.
     * @return The dot product as a Double.
     */
    public fun dot(a: T, b: T): Double

    // Tensor–Tensor
    /**
     * Adds two tensors element-wise.
     *
     * @param other The tensor to add.
     * @return The result of adding the two tensors.
     */
    public operator fun T.plus(other: T): T

    /**
     * Subtracts one tensor from another element-wise.
     *
     * @param other The tensor to subtract.
     * @return The result of subtracting the two tensors.
     */
    public operator fun T.minus(other: T): T

    /**
     * Multiplies two tensors element-wise.
     *
     * @param other The tensor to multiply with.
     * @return The result of multiplying the two tensors.
     */
    public operator fun T.times(other: T): T   // elementwise multiply

    /**
     * Divides one tensor by another element-wise.
     *
     * @param other The tensor to divide by.
     * @return The result of dividing the two tensors.
     */
    public operator fun T.div(other: T): T     // elementwise divide

    // Tensor–Scalar
    public operator fun T.plus(scalar: Int): T
    public operator fun T.minus(scalar: Int): T
    public operator fun T.times(scalar: Int): T
    public operator fun T.div(scalar: Int): T

    public operator fun T.plus(scalar: Float): T
    public operator fun T.minus(scalar: Float): T
    public operator fun T.times(scalar: Float): T
    public operator fun T.div(scalar: Float): T

    public operator fun T.plus(scalar: Double): T
    public operator fun T.minus(scalar: Double): T
    public operator fun T.times(scalar: Double): T
    public operator fun T.div(scalar: Double): T


    // Scalar–Tensor
    public operator fun Double.plus(t: T): T
    public operator fun Double.minus(t: T): T
    public operator fun Double.times(t: T): T
    public operator fun Double.div(t: T): T

    // Optionally Int & Float (delegate to Double)
    public operator fun Int.plus(t: T): T = this.toDouble() + t
    public operator fun Int.minus(t: T): T = this.toDouble() - t
    public operator fun Int.times(t: T): T = this.toDouble() * t
    public operator fun Int.div(t: T): T = this.toDouble() / t

    public operator fun Float.plus(t: T): T = this.toDouble() + t
    public operator fun Float.minus(t: T): T = this.toDouble() - t
    public operator fun Float.times(t: T): T = this.toDouble() * t
    public operator fun Float.div(t: T): T = this.toDouble() / t

    public fun T.t(): T  // transpose

    public fun T.relu(): T

    /**
     * Applies the softmax function along the specified dimension of the tensor.
     */
    public fun T.softmax(dimension: Int): T

    /**
     * Applies the sigmoid function element-wise to the tensor.
     *
     * @return A new tensor with the sigmoid function applied to each element.
     */
    public fun T.sigmoid(): T


    /**
     * Applies the hyperbolic tangent (tanh) function element-wise to the tensor.
     *
     * @return A new tensor with the tanh function applied to each element.
     */
    public fun T.tanh(): T


    /**
     * Flattens the tensor into a 1D tensor.
     *
     * @param startDim The first dimension to flatten (inclusive).
     * @param endDim The last dimension to flatten (inclusive).
     * @return A new flattened tensor.
     */
    public fun T.flatten(startDim: Int = 1, endDim: Int = -1): T

    /**
     * Reshapes the tensor to a new shape while preserving the total number of elements.
     * Similar to NumPy's reshape operation.
     *
     * @param newShape The new shape for the tensor. The total volume must match the original tensor's volume.
     * @return A new tensor with the specified shape containing the same data.
     * @throws IllegalArgumentException if the total volume of newShape doesn't match the original tensor's volume.
     */
    public fun T.reshape(newShape: Shape): T
    
    /**
     * Reshapes the tensor to a new shape while preserving the total number of elements.
     * Similar to NumPy's reshape operation with automatic dimension inference.
     * 
     * One dimension can be specified as -1, and it will be automatically calculated
     * based on the tensor's total volume and other specified dimensions.
     *
     * @param dimensions Variable number of dimensions. Exactly one can be -1 for automatic inference.
     * @return A new tensor with the inferred shape containing the same data.
     * @throws IllegalArgumentException if more than one dimension is -1 or if dimensions are invalid.
     */
    public fun T.reshape(vararg dimensions: Int): T
}


