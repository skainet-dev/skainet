package sk.ai.net.nn.topology

import sk.ai.net.Tensor
import sk.ai.net.nn.Module

class MLP(vararg modules: Module, override val name: String = "FeedForwardNetwork") : Module() {
    private val modulesList = modules.toList()
    override val modules: List<Module>
        get() = modulesList

    override fun forward(input: Tensor): Tensor {
        var tmp = input
        modulesList.forEach { module ->
            tmp = module.forward(tmp)
        }
        return tmp
    }
}