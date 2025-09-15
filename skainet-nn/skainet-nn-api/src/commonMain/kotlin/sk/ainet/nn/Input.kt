package sk.ainet.nn

import sk.ainet.core.tensor.DType
import sk.ainet.core.tensor.Shape
import sk.ainet.core.tensor.Tensor
import sk.ainet.core.tensor.TensorOps


public class Input<T : DType, V>(private val inputShape: Shape, override val name: String = "Input") : Module<T, V>() {

    override val modules: List<Module<T, V>>
        get() = emptyList()


    override fun TensorOps<Tensor<T, V>>.forward(input: Tensor<T, V>): Tensor<T, V> {
        return input
    }
}