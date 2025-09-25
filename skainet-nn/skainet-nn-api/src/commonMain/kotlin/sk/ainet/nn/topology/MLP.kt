package sk.ainet.nn.topology

import sk.ainet.core.tensor.DType
import sk.ainet.core.tensor.Tensor
import sk.ainet.nn.Module


public class MLP<T : DType, V>(vararg modules: Module<T, V>, override val name: String = "FeedForwardNetwork") :
    Module<T, V>(), ModuleParameters<T, V> {
    private val modulesList = modules.toList()
    override val modules: List<Module<T, V>>
        get() = modulesList

    override fun forward(input: Tensor<T, V>): Tensor<T, V> {
        var tmp = input
        modulesList.forEach { module ->
            tmp = module.forward(tmp)
        }
        return tmp
    }

    override val params: List<ModuleParameter<T, V>>
        get() = emptyList()
}