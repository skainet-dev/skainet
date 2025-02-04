package sk.ai.net.nn.reflection

import sk.ai.net.nn.Module

fun flattenParams(module: Module): List<ModuleParameter> {
    val params = mutableListOf<ModuleParameter>()
    for (m in module.modules) {
        params.addAll(flattenParams(m))
    }
    if (module is ModuleParameters) {
        params.addAll(module.params)
    }
    return params
}