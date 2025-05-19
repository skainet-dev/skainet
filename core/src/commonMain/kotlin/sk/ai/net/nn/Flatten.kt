package sk.ai.net.nn

import sk.ai.net.Tensor

class Flatten(
    private val startDim: Int = 1,
    private val endDim: Int = -1,
    override val name: String = "Flatten"
) : Module() {
    override val modules: List<Module>
        get() = emptyList()

    override fun forward(input: Tensor): Tensor {
        return input.flatten(startDim, endDim)
    }
}
