package sk.ainet.nn

import sk.ainet.core.tensor.DType
import sk.ainet.core.tensor.Tensor
import sk.ainet.core.tensor.TensorOps


public abstract class Module<T : DType, V> {

    public abstract val name: String

    public abstract val modules: List<Module<T, V>>

    public abstract fun TensorOps<Tensor<T, V>>.forward(input: Tensor<T, V>): Tensor<T, V>

    public operator fun TensorOps<Tensor<T, V>>.invoke(input: Tensor<T, V>): Tensor<T, V> {
        return forward(input)
    }
}

