package sk.ai.net.impl

import sk.ai.net.Shape
import sk.ai.net.Tensor

fun createTensor(values: DoubleArray): Tensor {
    return DoublesTensor(Shape(), values)
}

fun createTensor(values: IntArray): Tensor {
    return DoublesTensor(Shape(), values.map { it.toDouble() }.toDoubleArray())
}

fun createTensor(shape: Shape, values: DoubleArray): Tensor {
    return DefaultTensorFactory().createTensor(
        shape = shape,
        dataDescriptor = DefaultDataDescriptorFactory().createDataDescriptor(),
        elements = values
    )
}

fun createTensor(shape: Shape, values: IntArray): Tensor {
    return DefaultTensorFactory().createTensor(
        shape = shape,
        dataDescriptor = DefaultDataDescriptorFactory().createDataDescriptor(),
        elements = values.map { it.toDouble() }.toDoubleArray()
    )
}