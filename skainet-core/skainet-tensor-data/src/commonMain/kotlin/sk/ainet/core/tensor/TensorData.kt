package sk.ainet.core.tensor

public interface ItemsAccessor<T> {
    public operator fun get(index: Int): T
}

/**
 * Sealed interface hierarchy for TensorData implementations.
 * This ensures type-safe polymorphism for different data layouts.
 */
public sealed interface TensorData<T:DType, V>:ItemsAccessor<V> {
    /**
     * The shape of the tensor, represented as a Shape data class containing dimensions.
     *
     * For example, a scalar has an empty shape, a vector has a shape with one dimension,
     * a matrix has a shape with two dimensions, and so on.
     */
    public val shape: Shape

    /**
     * Creates a slice view of this tensor data.
     *
     * @param ranges The slice ranges for each dimension
     * @return A new TensorData representing the slice
     */
    public fun slice(ranges: IntArray): TensorData<T,V>

    /**
     * Materializes this tensor data into a contiguous representation.
     * If already contiguous, may return this same instance.
     *
     * @return A materialized (contiguous) version of this tensor data
     */
    public fun <V> materialize(): TensorData<T,V>
}