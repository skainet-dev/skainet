package sk.ainet.lang.nn.activations

import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.silu
import sk.ainet.lang.types.DType

public class SiLU<T : DType, V>(override val name: String = "SiLU") : Module<T, V>() {
    override val modules: List<Module<T, V>>
        get() = emptyList()

    override fun forward(input: Tensor<T, V>): Tensor<T, V> = with(input) { silu() }
}