package sk.ai.net.nn

import sk.ai.net.Tensor

data class ModuleParameter(
    val name: String,
    var value: Tensor
)

interface ModuleParameters {
    val params: List<ModuleParameter>
}

