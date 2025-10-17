package sk.ainet.lang.nn

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType


public class Input<T : DType, V>(override val name: String = "Input") : Module<T, V>() {

    override val modules: List<Module<T, V>>
        get() = emptyList()


    override fun forward(input: Tensor<T, V>): Tensor<T, V> {
        return input
    }
}