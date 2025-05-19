package sk.ai.net.nn

import sk.ai.net.Shape
import sk.ai.net.Tensor
import sk.ai.net.impl.DoublesTensor

/**
 * A simple layer that flattens an input tensor into a 1D tensor.
 * This layer has no parameters and simply reshapes the input.
 */
class Flatten(override val name: String = "Flatten") : Module() {
    override val modules: List<Module>
        get() = emptyList()

    override fun forward(input: Tensor): Tensor {
        val t = input as DoublesTensor
        return DoublesTensor(Shape(t.size), t.elements.copyOf())
    }
}
