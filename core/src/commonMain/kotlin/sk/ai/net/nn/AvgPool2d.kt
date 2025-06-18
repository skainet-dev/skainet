package sk.ai.net.nn

import sk.ai.net.Shape
import sk.ai.net.Tensor
import sk.ai.net.impl.DoublesTensor

/**
 * 2D average pooling layer.
 * Works with tensors of shape (N, C, H, W) or (C, H, W).
 */
class AvgPool2d(
    val kernelSize: Int,
    val stride: Int = kernelSize,
    override val name: String = "AvgPool2d"
) : Module() {
    override val modules: List<Module>
        get() = emptyList()

    override fun forward(input: Tensor): Tensor = avgPool2d(input)

    private fun avgPool2d(input: Tensor): Tensor {
        val tensor = input as DoublesTensor
        val shape = tensor.shape
        require(shape.rank == 3 || shape.rank == 4) {
            "AvgPool2d expected 3D or 4D input tensor, but got shape $shape"
        }
        val batchSize: Int
        val channels: Int
        val height: Int
        val width: Int
        if (shape.rank == 4) {
            batchSize = shape.dimensions[0]
            channels = shape.dimensions[1]
            height = shape.dimensions[2]
            width = shape.dimensions[3]
        } else {
            batchSize = 1
            channels = shape.dimensions[0]
            height = shape.dimensions[1]
            width = shape.dimensions[2]
        }

        val outH = (height - kernelSize) / stride + 1
        val outW = (width - kernelSize) / stride + 1
        val outElements = DoubleArray(batchSize * channels * outH * outW)
        var idx = 0
        for (n in 0 until batchSize) {
            for (c in 0 until channels) {
                for (i in 0 until outH) {
                    for (j in 0 until outW) {
                        var sum = 0.0
                        for (ki in 0 until kernelSize) {
                            for (kj in 0 until kernelSize) {
                                val h = i * stride + ki
                                val w = j * stride + kj
                                val value = if (shape.rank == 4) {
                                    tensor.elements[n * channels * height * width + c * height * width + h * width + w]
                                } else {
                                    tensor.elements[c * height * width + h * width + w]
                                }
                                sum += value
                            }
                        }
                        outElements[idx++] = sum / (kernelSize * kernelSize)
                    }
                }
            }
        }
        val outShape = Shape(batchSize, channels, outH, outW)
        return DoublesTensor(outShape, outElements)
    }
}