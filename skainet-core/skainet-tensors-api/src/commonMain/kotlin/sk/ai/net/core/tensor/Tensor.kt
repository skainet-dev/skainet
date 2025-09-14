package sk.ai.net.core.tensor

/**
 * Interface representing a multi-dimensional array of numeric values.
 *
 * A Tensor is a generalization of vectors and matrices to potentially higher dimensions.
 * The [shape] property defines the dimensions of the tensor, and the [get] operator
 * allows accessing individual elements by their indices.
 */
public interface Tensor<T : DType, V> : TensorData<T, V>, TensorOps<Tensor<T, V>>
