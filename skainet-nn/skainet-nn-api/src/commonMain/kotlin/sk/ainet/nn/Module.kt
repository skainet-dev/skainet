package sk.ainet.nn

import sk.ainet.core.tensor.DType
import sk.ainet.core.tensor.Tensor


public abstract class Module<T : DType, V> {

    public abstract val name: String

    public abstract val modules: List<Module<T, V>>

    public abstract fun forward(input: Tensor<T, V>): Tensor<T, V>

    public operator fun invoke(input: Tensor<T, V>): Tensor<T, V> {
        return forward(input)
    }
}

