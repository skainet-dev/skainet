package sk.ainet.lang.tensor.ops

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType


public interface TensorOps<V> {
    // Basic mathematical operations
    public fun <T : DType> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>
    public fun <T : DType> subtract(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>
    public fun <T : DType> multiply(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>
    public fun <T : DType> divide(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>
    
    // Linear algebra operations
    public fun <T : DType> matmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>
    public fun <T : DType> transpose(tensor: Tensor<T, V>): Tensor<T, V>
    
    // Convolutional operations
    public fun <T : DType> conv2d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>? = null,
        stride: Pair<Int, Int> = 1 to 1,
        padding: Pair<Int, Int> = 0 to 0,
        dilation: Pair<Int, Int> = 1 to 1,
        groups: Int = 1
    ): Tensor<T, V>
    
    // Pooling operations
    public fun <T : DType> maxPool2d(
        input: Tensor<T, V>,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int> = kernelSize,
        padding: Pair<Int, Int> = 0 to 0
    ): Tensor<T, V>
    
    // Shape operations
    public fun <T : DType> reshape(tensor: Tensor<T, V>, newShape: Shape): Tensor<T, V>
    public fun <T : DType> flatten(tensor: Tensor<T, V>, startDim: Int = 0, endDim: Int = -1): Tensor<T, V>
    
    // Activation functions
    public fun <T : DType> relu(tensor: Tensor<T, V>): Tensor<T, V>
    public fun <T : DType> softmax(tensor: Tensor<T, V>, dim: Int = -1): Tensor<T, V>
    public fun <T : DType> sigmoid(tensor: Tensor<T, V>): Tensor<T, V>
    public fun <T : DType> silu(tensor: Tensor<T, V>): Tensor<T, V>
    public fun <T : DType> gelu(tensor: Tensor<T, V>): Tensor<T, V>
    
    // Reduction operations
    public fun <T : DType> sum(tensor: Tensor<T, V>, dim: Int? = null): Tensor<T, V>
    public fun <T : DType> mean(tensor: Tensor<T, V>, dim: Int? = null): Tensor<T, V>
    public fun <T : DType> variance(tensor: Tensor<T, V>, dim: Int? = null): Tensor<T, V>
    
    // Mathematical functions
    public fun <T : DType> sqrt(tensor: Tensor<T, V>): Tensor<T, V>
    
    // Type conversion operations
    public fun <TFrom : DType, TTo : DType> convert(
        tensor: Tensor<TFrom, V>, 
        targetType: TTo
    ): Tensor<TTo, V>
}