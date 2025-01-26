package sk.ai.net

/**
 * An interface to describe tensor data type. An internar representation of a tensor can differ
 * from the {@see DataDescriptor} data type. It provides an access the tensor data via getter method using built-in
 * types(Int, Float etc).
 */
interface Tensor<T> {

    /**
     * shape indicates the number of dimension of a tensor and the length of each dimension.
     * If the shape of the tensor is 3x4x5 then the value of shape is Shape(3,4,5).
     * */
    val shape: Shape

    /**
     * The number of dimensions in the tensor's shape.
     * rank 0 - Scalar
     *  rank 1 - 1D array or 1D tensor
     *  rank 2 - 2D matrix or 2D tensor
     *  rank 3 - 3D tensor
     *  ...
     *  rank N - ND tensor
     *  */
    val rank: Int get() = shape.rank

    /**
     * The total number of elements of this tensor.
     * */
    val size: Int get() = shape.volume

    /**
     * The uniform data descriptor of the tensor.
     * */
    val dataDescriptor: DataDescriptor

    operator fun get(vararg indices: Int): T

    operator fun get(vararg ranges: IntRange): Tensor<T>
}

fun <T> Tensor<T>.isScalar() = rank == 0

fun <T> Tensor<T>.isVector() = rank == 1

