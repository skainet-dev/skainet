package sk.ainet.lang.nn.activations

import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.gelu
import sk.ainet.lang.types.DType

public class GELU<T : DType, V>(override val name: String = "GELU") : Module<T, V>() {
    override val modules: List<Module<T, V>>
        get() = emptyList()

    override fun forward(input: Tensor<T, V>): Tensor<T, V> = with(input) { gelu() }
}