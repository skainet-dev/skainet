package sk.ainet.lang.tensor.ops

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

public class VoidTensorOps<V> : TensorOps<V> {
    override fun <T : DType> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> subtract(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> multiply(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> divide(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> matmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> transpose(tensor: Tensor<T, V>): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> reshape(tensor: Tensor<T, V>, newShape: Shape): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> flatten(tensor: Tensor<T, V>, startDim: Int, endDim: Int): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> relu(tensor: Tensor<T, V>): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> softmax(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> sigmoid(tensor: Tensor<T, V>): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> sum(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType> mean(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun <TFrom : DType, TTo : DType> convert(
        tensor: Tensor<TFrom, V>,
        targetType: TTo
    ): Tensor<TTo, V> {
        TODO("Not yet implemented")
    }
}
