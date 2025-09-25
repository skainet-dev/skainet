package sk.ainet.nn

import sk.ainet.core.tensor.DType
import sk.ainet.core.tensor.Tensor
import sk.ainet.nn.topology.ModuleParameter
import sk.ainet.nn.topology.ModuleParameters
import sk.ainet.nn.topology.bias
import sk.ainet.nn.topology.weights

/**
 * Linear layer (a.k.a. fully connected dense layer). This layer applies a linear transformation to the input data.
 * The weights and biases are learned during training.
 *
 * @param inFeatures Number of input features
 * @param outFeatures Number of output features
 * @param name Name of the module
 * @param initWeights Initial weights
 * @param initBias Initial bias
 */

public class Linear<T : DType, V>(
    inFeatures: Int,
    outFeatures: Int,
    override val name: String = "Linear",
    initWeights: Tensor<T, V>, //= Tensor.randn<T, V>(shape = intArrayOf(outFeatures, inFeatures)),
    initBias: Tensor<T, V>, //  = Tensor.zeros<T, V>(shape = intArrayOf(outFeatures)),
) : Module<T, V>(), ModuleParameters<T, V> {
    override val params: List<ModuleParameter<T, V>> = listOf(
        ModuleParameter.WeightParameter("$name.weight", initWeights),
        ModuleParameter.BiasParameter("$name.bias", initBias)
    )


    override val modules: List<Module<T, V>>
        get() = emptyList()

    override fun forward(input: Tensor<T, V>): Tensor<T, V> {
        val weight = params.weights().value
        val bias = params.bias().value

        // Use TensorOps context operations
        with(input) {
            val weightTransposed = weight.t()
            val matmulResult = matmul(input, weightTransposed)
            return matmulResult + bias
        }
    }
}
