package sk.ai.net.impl

import sk.ai.net.Shape
import sk.ai.net.Tensor

fun createTensor(values: DoubleArray): Tensor {
    return DoublesTensor(Shape(), values)
}

fun createTensor(shape: Shape, values: DoubleArray): Tensor {
    return DoublesTensor(shape, values)
}