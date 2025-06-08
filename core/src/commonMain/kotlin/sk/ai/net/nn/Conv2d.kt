package sk.ai.net.nn

import sk.ai.net.Shape
import sk.ai.net.Tensor
import sk.ai.net.impl.DoublesTensor
import sk.ai.net.rand
import sk.ai.net.zeros
import kotlin.math.sqrt

class Conv2d(
    val inChannels: Int,
    val outChannels: Int,
    val kernelSize: Int,
    val stride: Int = 1,
    val padding: Int = 0,
    useBias: Boolean = true,
    name: String = "Conv2d"
) : Module() {
    override val name: String = name
    val weight: Tensor
    val bias: Tensor?
    override val modules: List<Module>
        get() = emptyList()

    override fun forward(input: Tensor): Tensor = con2d(input)

    init {
        // Initialize weights and bias
        val fanIn = inChannels * kernelSize * kernelSize
        val bound = 1.0 / sqrt(fanIn.toDouble())  // 1/sqrt(fanIn)

        // Weight: uniform in [-bound, bound]
        // Create a tensor with random values in [0,1]
        val weightShape = Shape(outChannels, inChannels, kernelSize, kernelSize)
        val randomWeight = rand(weightShape) as DoublesTensor

        // Scale to [0, 2*bound]
        val scaledWeight = randomWeight.times(2.0 * bound)

        // Shift to [-bound, bound]
        val boundTensor = DoublesTensor(weightShape, DoubleArray(weightShape.volume) { bound })
        weight = scaledWeight.minus(boundTensor)

        // Bias: uniform in [-bound, bound] if enabled
        bias = if (useBias) {
            val biasShape = Shape(outChannels)
            val randomBias = rand(biasShape) as DoublesTensor
            val scaledBias = randomBias.times(2.0 * bound)
            val boundBiasTensor = DoublesTensor(biasShape, DoubleArray(biasShape.volume) { bound })
            scaledBias.minus(boundBiasTensor)
        } else {
            null
        }
    }

    fun con2d(input: Tensor): Tensor {
        // Ensure input has 3D or 4D shape
        val shape = input.shape  // assume shape is a list or array of dimensions
        require(shape.rank == 3 || shape.rank == 4) {
            "Conv2d expected 3D or 4D input tensor, but got shape ${shape}."
        }
        // Determine batch size and input dims
        val batchSize: Int
        val inC: Int
        val inH: Int
        val inW: Int
        if (shape.rank == 4) {
            batchSize = shape.dimensions[0]
            inC = shape.dimensions[1]
            inH = shape.dimensions[2]
            inW = shape.dimensions[3]
        } else {
            // if 3D (C, H, W), treat as batch of size 1
            batchSize = 1
            inC = shape.dimensions[0]
            inH = shape.dimensions[1]
            inW = shape.dimensions[2]
        }
        require(inC == inChannels) {
            "Conv2d expected input channel count $inChannels, but got $inC."
        }

        // Compute output spatial dimensions
        val outH = (inH + 2 * padding - kernelSize) / stride + 1
        val outW = (inW + 2 * padding - kernelSize) / stride + 1
        require(outH > 0 && outW > 0) {
            "Conv2d output size is invalid (outH=$outH, outW=$outW). Check input dimensions and padding."
        }

        // Cast input to DoublesTensor for easier access
        val inputTensor = input as DoublesTensor

        // Apply padding if needed
        val paddedInput: DoublesTensor = if (padding > 0) {
            val paddedH = inH + 2 * padding
            val paddedW = inW + 2 * padding
            val paddedSize = batchSize * inC * paddedH * paddedW
            val paddedElements = DoubleArray(paddedSize) { 0.0 }

            // Calculate strides for the padded tensor
            val paddedStrides = IntArray(4)
            paddedStrides[3] = 1
            paddedStrides[2] = paddedW
            paddedStrides[1] = paddedH * paddedW
            paddedStrides[0] = inC * paddedH * paddedW

            // Fill the padded tensor with the input values
            for (n in 0 until batchSize) {
                for (c in 0 until inC) {
                    for (i in 0 until inH) {
                        for (j in 0 until inW) {
                            val paddedIdx = n * paddedStrides[0] + c * paddedStrides[1] + 
                                           (i + padding) * paddedStrides[2] + (j + padding) * paddedStrides[3]

                            val inputValue = if (shape.rank == 4) {
                                inputTensor[n, c, i, j]
                            } else {
                                inputTensor[c, i, j]
                            }

                            paddedElements[paddedIdx] = inputValue
                        }
                    }
                }
            }

            DoublesTensor(Shape(batchSize, inC, paddedH, paddedW), paddedElements)
        } else {
            inputTensor  // no padding needed
        }

        // Prepare output tensor
        val outputElements = DoubleArray(batchSize * outChannels * outH * outW)
        var idx = 0

        // Cast weight and bias to DoublesTensor for easier access
        val weightTensor = weight as DoublesTensor
        val biasTensor = bias as? DoublesTensor

        // Convolution: iterate over batch, out channels, and output spatial positions
        for (n in 0 until batchSize) {
            for (oc in 0 until outChannels) {
                val biasVal = if (biasTensor != null) biasTensor[oc] else 0.0
                for (i in 0 until outH) {
                    for (j in 0 until outW) {
                        var sum = 0.0
                        // Sum over all input channels and kernel elements
                        for (c in 0 until inChannels) {
                            for (ki in 0 until kernelSize) {
                                for (kj in 0 until kernelSize) {
                                    val h = i * stride + ki
                                    val w = j * stride + kj
                                    sum += paddedInput[n, c, h, w] * weightTensor[oc, c, ki, kj]
                                }
                            }
                        }
                        // Add bias and assign to output
                        outputElements[idx++] = sum + biasVal
                    }
                }
            }
        }

        return DoublesTensor(Shape(batchSize, outChannels, outH, outW), outputElements)
    }
}
