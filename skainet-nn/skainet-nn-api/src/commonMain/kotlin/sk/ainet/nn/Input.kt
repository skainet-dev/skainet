package sk.ainet.nn

import sk.ainet.core.tensor.DType
import sk.ainet.core.tensor.Tensor


public class Input<T : DType, V>(override val name: String = "Input") : Module<T, V>() {

    override val modules: List<Module<T, V>>
        get() = emptyList()


    override fun forward(input: Tensor<T, V>): Tensor<T, V> {
        return input
    }
}