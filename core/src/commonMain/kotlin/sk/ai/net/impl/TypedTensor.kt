package sk.ai.net.impl

import sk.ai.net.Tensor

/**
 * An interface to a tensor with elements of a specific data type. An internar representation of a tensor can differ
 * from the {@see DataDescriptor} data type. It provides an access the tensor data via getter method using built-in
 * types(Int, Float etc).
 */
interface TypedTensor<T> : Tensor {
    operator fun get(vararg indices: Int): T

    operator fun get(vararg ranges: IntRange): TypedTensor<T>
}



