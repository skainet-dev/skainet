package sk.ai.net.impl

import sk.ai.net.Shape
import sk.ai.net.Tensor

fun createTensor(values: DoubleArray): Tensor<Double> {
    return DoublesTensor(Shape(), values)
}

fun createTensor(shape: Shape, values: DoubleArray): Tensor<Double> {
    return DoublesTensor(shape, values)
}