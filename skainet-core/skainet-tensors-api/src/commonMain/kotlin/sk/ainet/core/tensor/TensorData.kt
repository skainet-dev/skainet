package sk.ainet.core.tensor

/**
 * Sealed interface hierarchy for TensorData implementations.
 * This ensures type-safe polymorphism for different data layouts.
 */
public sealed interface TensorData<T:DType, V> {
    /**
     * The shape of the tensor, represented as a Shape data class containing dimensions.
     *
     * For example, a scalar has an empty shape, a vector has a shape with one dimension,
     * a matrix has a shape with two dimensions, and so on.
     */
    public val shape: Shape

    /**
     * Stride information for each dimension.
     * Represents the step size in memory for each dimension.
     */
    public val strides: IntArray

    /**
     * Offset in the underlying data array where this tensor data begins.
     */
    public val offset: Int

    /**
     * Whether this tensor data represents contiguous data in memory.
     * Contiguous data can be optimized for better performance.
     */
    public val isContiguous: Boolean

    /**
     * Retrieves the value at the specified indices.
     *
     * @param indices The indices of the element to retrieve.
     * @return The value at the specified indices.
     */
    public operator fun get(vararg indices: Int): V

    /**
     * Copies the tensor data to the destination array.
     *
     * @param dest The destination array
     * @param destOffset The offset in the destination array
     */
    public fun copyTo(dest: Array<V>, destOffset: Int = 0)

    /**
     * Creates a slice view of this tensor data.
     *
     * @param ranges The slice ranges for each dimension
     * @return A new TensorData representing the slice
     */
    public fun slice(ranges: IntArray): TensorData<T, V>

    /**
     * Materializes this tensor data into a contiguous representation.
     * If already contiguous, may return this same instance.
     *
     * @return A materialized (contiguous) version of this tensor data
     */
    public fun materialize(): TensorData<T, V>
}