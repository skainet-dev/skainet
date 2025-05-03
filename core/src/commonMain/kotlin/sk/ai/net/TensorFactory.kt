package sk.ai.net

import sk.ai.net.impl.BuiltInDoubleDataDescriptor
import sk.ai.net.impl.DoublesTensor
import kotlin.random.Random

interface TensorFactory {
    fun createTensor(shape: Shape, dataDescriptor: DataDescriptor, elements: DoubleArray): Tensor
}

fun rand(shape: Shape, dataDescriptor: DataDescriptor = BuiltInDoubleDataDescriptor()): Tensor {
    val random: Random = Random.Default

    return DoublesTensor(shape, DoubleArray(shape.volume) { random.nextFloat().toDouble() })
}

fun zeros(shape: Shape, dataDescriptor: DataDescriptor = BuiltInDoubleDataDescriptor()): Tensor {
    return DoublesTensor(shape, DoubleArray(shape.volume))
}
