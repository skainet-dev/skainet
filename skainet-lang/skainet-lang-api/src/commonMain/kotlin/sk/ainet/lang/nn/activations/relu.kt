package sk.ainet.lang.nn.activations

import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.relu
import sk.ainet.lang.types.DType

public class ReLU<T : DType, V>(override val name: String = "ReLU") : Module<T, V>() {
    override val modules: List<Module<T, V>>
        get() = emptyList()

    override fun forward(input: Tensor<T, V>): Tensor<T, V> = with(input) { relu() }
}

