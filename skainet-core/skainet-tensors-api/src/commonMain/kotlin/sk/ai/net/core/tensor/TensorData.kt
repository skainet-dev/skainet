package sk.ai.net.core.tensor

public interface TensorData<T:DType, V> {
    /**
     * The shape of the tensor, represented as a Shape data class containing dimensions.
     *
     * For example, a scalar has an empty shape, a vector has a shape with one dimension,
     * a matrix has a shape with two dimensions, and so on.
     */
    public val shape: Shape

    /**
     * Retrieves the value at the specified indices.
     *
     * @param indices The indices of the element to retrieve.
     * @return The value at the specified indices.
     */
    public operator fun get(vararg indices: Int): V
}