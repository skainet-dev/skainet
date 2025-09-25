package sk.ainet.nn.activations

import sk.ainet.core.tensor.DType
import sk.ainet.core.tensor.Tensor
import sk.ainet.nn.Module

public class Softmax<T : DType, V>(private val dimension: Int, override val name: String = "Softmax") : Module<T, V>() {
    override val modules: List<Module<T, V>>
        get() = emptyList()

    override fun forward(input: Tensor<T, V>): Tensor<T, V> {
        return with(input) {
            softmax(dimension)
        }
    }
}

