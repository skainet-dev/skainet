package sk.ai.net

interface TensorFactory {
    fun createTensor(shape: Shape, dataDescriptor: DataDescriptor, elements: DoubleArray): Tensor
}