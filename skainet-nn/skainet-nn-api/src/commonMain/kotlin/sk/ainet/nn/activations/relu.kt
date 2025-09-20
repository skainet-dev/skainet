package sk.ainet.nn.activations

import sk.ainet.core.tensor.DType
import sk.ainet.core.tensor.Tensor
import sk.ainet.core.tensor.TensorOps
import sk.ainet.nn.Module

public class ReLU<T : DType, V>(override val name: String = "ReLU") : Module<T, V>() {
    override val modules: List<Module<T, V>>
        get() = emptyList()

    override fun TensorOps<Tensor<T, V>>.forward(input: Tensor<T, V>): Tensor<T, V> = input.relu()
}

