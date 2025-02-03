package sk.ai.net.impl

import sk.ai.net.DataDescriptor
import sk.ai.net.Shape
import sk.ai.net.Tensor
import sk.ai.net.TensorFactory

class DefaultTensorFactory : TensorFactory {
    override fun createTensor(shape: Shape, dataDescriptor: DataDescriptor, elements: DoubleArray): Tensor {
        if (dataDescriptor.isFloatingPoint) {
            return DoublesTensor(shape, elements)
        } else {
            throw IllegalArgumentException("Unsupported data descriptor type: $dataDescriptor")
        }
    }
}