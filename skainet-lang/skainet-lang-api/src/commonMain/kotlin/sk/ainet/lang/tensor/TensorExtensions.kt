package sk.ainet.lang.tensor

import sk.ainet.lang.types.DType

// Tensor extension functions that delegate to the ops component
public fun <T : DType, V> Tensor<T, V>.t(): Tensor<T, V> = ops.transpose(this)
public fun <T : DType, V> Tensor<T, V>.matmul(other: Tensor<T, V>): Tensor<T, V> = ops.matmul(this, other)
public fun <T : DType, V> Tensor<T, V>.flatten(startDim: Int = 0, endDim: Int = -1): Tensor<T, V> = 
    ops.flatten(this, startDim, endDim)

// Operator overloads
public operator fun <T : DType, V> Tensor<T, V>.plus(other: Tensor<T, V>): Tensor<T, V> = ops.add(this, other)
public operator fun <T : DType, V> Tensor<T, V>.minus(other: Tensor<T, V>): Tensor<T, V> = ops.subtract(this, other)
public operator fun <T : DType, V> Tensor<T, V>.times(other: Tensor<T, V>): Tensor<T, V> = ops.multiply(this, other)
public operator fun <T : DType, V> Tensor<T, V>.div(other: Tensor<T, V>): Tensor<T, V> = ops.divide(this, other)

// Additional convenience functions
public fun <T : DType, V> Tensor<T, V>.reshape(newShape: Shape): Tensor<T, V> = ops.reshape(this, newShape)
public fun <T : DType, V> Tensor<T, V>.relu(): Tensor<T, V> = ops.relu(this)
public fun <T : DType, V> Tensor<T, V>.sigmoid(): Tensor<T, V> = ops.sigmoid(this)
public fun <T : DType, V> Tensor<T, V>.softmax(dim: Int = -1): Tensor<T, V> = ops.softmax(this, dim)
public fun <T : DType, V> Tensor<T, V>.sum(dim: Int? = null): Tensor<T, V> = ops.sum(this, dim)
public fun <T : DType, V> Tensor<T, V>.mean(dim: Int? = null): Tensor<T, V> = ops.mean(this, dim)

// Global matmul function for the Linear layer usage pattern
public fun <T : DType, V> matmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> = a.ops.matmul(a, b)