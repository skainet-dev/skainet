package sk.ai.net.nn

import sk.ai.net.Shape
import sk.ai.net.Tensor


class Input(private val inputShape: Shape, override val name: String = "Input") : Module() {

    override val modules: List<Module>
        get() = emptyList()


    override fun forward(input: Tensor): Tensor {
        return input
    }
}