package sk.ai.net.nn

import sk.ai.net.Shape
import sk.ai.net.Tensor
import sk.ai.net.impl.DoublesTensor

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

class Linear(
    inFeatures: Int,
    outFeatures: Int,
    override val name: String = "Linear",
    val initWeights: Tensor = DoublesTensor(
        Shape(outFeatures, inFeatures),
        List(inFeatures * outFeatures) { 0.0 }.map { it }.toDoubleArray()
    ),
    val initBias: Tensor = DoublesTensor(
        Shape(outFeatures),
        List(outFeatures) { 0.0 }.map { it }.toDoubleArray()
    ),
    override val params: List<ModuleParameter> = listOf(
        ModuleParameter("weight", initWeights),
        ModuleParameter("bias", initBias)
    ),
) : Module(), ModuleParameters {

    override val modules: List<Module>
        get() = emptyList()

    override fun forward(input: Tensor): Tensor {
        val weight = initWeights
        val bias = initBias

        // matrix multiplication on tensors and addition
        return input.matmul(weight.t()) + bias
    }
}
