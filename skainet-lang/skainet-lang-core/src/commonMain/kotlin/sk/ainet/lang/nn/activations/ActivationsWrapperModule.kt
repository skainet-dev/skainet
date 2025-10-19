package sk.ainet.lang.nn.activations

import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType


public class ActivationsWrapperModule<T : DType, V>(
    private val activationHandler: (Tensor<T, V>) -> Tensor<T, V>,
    override val name: String
) :
    Module<T, V>() {
    override val modules: List<Module<T, V>>
        get() = emptyList()

    override fun forward(input: Tensor<T, V>): Tensor<T, V> {
        return activationHandler(input)
    }
}