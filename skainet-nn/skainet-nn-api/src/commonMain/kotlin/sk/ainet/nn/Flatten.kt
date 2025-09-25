package sk.ainet.nn

import sk.ainet.core.tensor.DType
import sk.ainet.core.tensor.Tensor

/**
 * A simple layer that flattens an input tensor into a 1D tensor.
 * This layer has no parameters and simply reshapes the input.
 */
public class Flatten<T : DType, V>(
    private val startDim: Int = 1,
    private val endDim: Int = -1,
    override val name: String = "Flatten"
) : Module<T, V>() {
    override val modules: List<Module<T, V>>
        get() = emptyList()

    override fun forward(input: Tensor<T, V>): Tensor<T, V> {
        return with(input) {
            flatten(startDim, endDim)
        }
    }
}
