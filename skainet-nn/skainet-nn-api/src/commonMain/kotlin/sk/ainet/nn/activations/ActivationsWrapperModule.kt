package sk.ainet.nn.activations

import sk.ainet.core.tensor.DType
import sk.ainet.core.tensor.Tensor
import sk.ainet.core.tensor.TensorOps
import sk.ainet.nn.Module


public class ActivationsWrapperModule<T : DType, V>(
    private val activationHandler: (Tensor<T, V>) -> Tensor<T, V>,
    override val name: String
) :
    Module<T, V>() {
    override val modules: List<Module<T, V>>
        get() = emptyList()

    override fun Tensor<T, V>.forward(input: Tensor<T, V>): Tensor<T, V> {
        return activationHandler(input)
    }
}