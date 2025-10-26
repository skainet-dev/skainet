package sk.ainet.lang.tensor.ops

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Extension functions to make tensor operations available in the graph execution context.
 * These operators automatically use the current graph execution context when available.
 */

/**
 * Addition operator that works within graph execution context
 */
public operator fun <T: DType, V> Tensor<T, V>.plus(other: Tensor<T, V>): Tensor<T, V> {
    val context = getCurrentGraphContext<T, V>()
    return context.graphOps.add(this, other)
}

/**
 * Subtraction operator that works within graph execution context
 */
public operator fun <T: DType, V> Tensor<T, V>.minus(other: Tensor<T, V>): Tensor<T, V> {
    val context = getCurrentGraphContext<T, V>()
    return context.graphOps.subtract(this, other)
}

/**
 * Multiplication operator that works within graph execution context
 */
public operator fun <T: DType, V> Tensor<T, V>.times(other: Tensor<T, V>): Tensor<T, V> {
    val context = getCurrentGraphContext<T, V>()
    return context.graphOps.multiply(this, other)
}

/**
 * Division operator that works within graph execution context
 */
public operator fun <T: DType, V> Tensor<T, V>.div(other: Tensor<T, V>): Tensor<T, V> {
    val context = getCurrentGraphContext<T, V>()
    return context.graphOps.divide(this, other)
}

/**
 * Matrix multiplication that works within graph execution context
 */
public fun <T: DType, V> Tensor<T, V>.matmul(other: Tensor<T, V>): Tensor<T, V> {
    val context = getCurrentGraphContext<T, V>()
    return context.graphOps.matmul(this, other)
}

/**
 * ReLU activation that works within graph execution context
 */
public fun <T: DType, V> Tensor<T, V>.relu(): Tensor<T, V> {
    val context = getCurrentGraphContext<T, V>()
    return context.graphOps.relu(this)
}

/**
 * Softmax activation that works within graph execution context
 */
public fun <T: DType, V> Tensor<T, V>.softmax(dim: Int = -1): Tensor<T, V> {
    val context = getCurrentGraphContext<T, V>()
    return context.graphOps.softmax(this, dim)
}