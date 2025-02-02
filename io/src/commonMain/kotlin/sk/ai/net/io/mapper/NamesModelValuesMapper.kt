package sk.ai.net.io.mapper

import sk.ai.net.Tensor
import sk.ai.net.nn.Module
import sk.ai.net.nn.ModuleParameters

class NamesBasedValuesModelMapper(
    private val weightKeyRule: (Int) -> String = { index -> "fc${index + 1}.weight" },
    private val biasKeyRule: (Int) -> String = { index -> "fc${index + 1}.bias" }
) : ModelValuesMapper {

    override fun mapToModel(model: Module, wandb: Map<String, Tensor>) {

        if (model is ModuleParameters) {
            val linear = model.modules.filter { module ->
                module.name.startsWith("Li")
            }
            // weights
            linear.forEachIndexed { index, layer ->
                if (layer is ModuleParameters) {
                    val weightKey = weightKeyRule(index)
                    layer.params.firstOrNull { it.name.startsWith("w") }?.let { weightParam ->
                        wandb[weightKey]?.let { newWeightValue ->
                            weightParam.value = newWeightValue
                        }
                    }
                    val biasKey = biasKeyRule(index)
                    layer.params.firstOrNull { it.name.startsWith("b") }?.let { biasParam ->
                        wandb[biasKey]?.let { newWeightValue ->
                            biasParam.value = newWeightValue
                        }
                    }
                }
            }
        }
    }
}
