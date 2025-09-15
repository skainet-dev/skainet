package sk.ai.net.core.tensor

/**
 * Interface representing mathematical operations on tensors.
 *
 * Keeps math separate from storage, so the same math API can work
 * with multiple backends (dense, sparse, GPU, etc.).
 */
public interface  TensorOps<T: Tensor<*, *>> {
    /**
     * Performs matrix multiplication of two tensors.
     *
     * @param a The first tensor.
     * @param b The second tensor.
     * @return The result of matrix multiplication.
     */
    public fun matmul(a: T, b: T): T

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
}