package sk.ai.net.nn.reflection

import sk.ai.net.Tensor

data class ModuleParameter(
    val name: String,
    var value: Tensor
)

interface ModuleParameters {
    val params: List<ModuleParameter>
}

public fun List<ModuleParameter>.by(name: String): ModuleParameter? =
    firstOrNull { namedParameter -> namedParameter.name.uppercase().startsWith(name.uppercase()) }


