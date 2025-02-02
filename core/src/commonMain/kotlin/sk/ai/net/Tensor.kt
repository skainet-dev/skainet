package sk.ai.net

/**
 * An interface to a tensor data type. Its provide operations method signatures for tensor manipulation.
 */
interface Tensor {

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

    operator fun plus(other: Tensor): Tensor

    operator fun plus(other: Double): Tensor

    operator fun plus(other: Int): Tensor

    operator fun minus(other: Double): Tensor

    operator fun minus(other: Int): Tensor

    operator fun minus(other: Tensor): Tensor

    fun matmul(other: Tensor): Tensor

    fun t(): Tensor

    fun relu(): Tensor

    fun softmax(i: Int): Tensor

    fun softmax(): Tensor

    fun pow(tensor: Tensor): Tensor

    fun pow(scalar: Double): Tensor

    fun sin(): Tensor

    fun cos(): Tensor

    fun tan(): Tensor
    
    fun asin(): Tensor

    fun acos(): Tensor
    
    fun atan(): Tensor

    fun sinh():Tensor

    fun cosh():Tensor

    fun tanh():Tensor

    fun exp():Tensor

    fun log():Tensor

    fun sqrt():Tensor

    fun cbrt():Tensor

    fun sigmoid():Tensor

    fun ln():Tensor

}

