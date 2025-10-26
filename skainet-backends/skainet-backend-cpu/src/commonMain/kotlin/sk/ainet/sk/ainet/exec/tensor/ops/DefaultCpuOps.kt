package sk.ainet.sk.ainet.exec.tensor.ops

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.DType

public class DefaultCpuOps<V> : TensorOps<V> {
    override fun <T : DType> add(
        a: Tensor<T, V>,
        b: Tensor<T, V>
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> subtract(
        a: Tensor<T, V>,
        b: Tensor<T, V>
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> multiply(
        a: Tensor<T, V>,
        b: Tensor<T, V>
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> divide(
        a: Tensor<T, V>,
        b: Tensor<T, V>
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> matmul(
        a: Tensor<T, V>,
        b: Tensor<T, V>
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> transpose(tensor: Tensor<T, V>): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> conv2d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        dilation: Pair<Int, Int>,
        groups: Int
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> maxPool2d(
        input: Tensor<T, V>,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> reshape(
        tensor: Tensor<T, V>,
        newShape: Shape
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> flatten(
        tensor: Tensor<T, V>,
        startDim: Int,
        endDim: Int
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> concat(
        tensors: List<Tensor<T, V>>,
        dim: Int
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> split(
        tensor: Tensor<T, V>,
        splitSize: Int,
        dim: Int
    ): List<Tensor<T, V>> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> squeeze(
        tensor: Tensor<T, V>,
        dim: Int?
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> unsqueeze(
        tensor: Tensor<T, V>,
        dim: Int
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> relu(tensor: Tensor<T, V>): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> softmax(
        tensor: Tensor<T, V>,
        dim: Int
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> sigmoid(tensor: Tensor<T, V>): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> silu(tensor: Tensor<T, V>): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> gelu(tensor: Tensor<T, V>): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> sum(
        tensor: Tensor<T, V>,
        dim: Int?
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> mean(
        tensor: Tensor<T, V>,
        dim: Int?
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> variance(
        tensor: Tensor<T, V>,
        dim: Int?
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> sqrt(tensor: Tensor<T, V>): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <TFrom : DType, TTo : DType> convert(
        tensor: Tensor<TFrom, V>,
        targetType: TTo
    ): Tensor<TTo, V> {
        TODO("Not yet implemented")
    }

}