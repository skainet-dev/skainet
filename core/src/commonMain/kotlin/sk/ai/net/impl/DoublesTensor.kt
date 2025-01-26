package sk.ai.net.impl

import sk.ai.net.DataDescriptor
import sk.ai.net.Shape
import sk.ai.net.Tensor

data class DoublesTensor(override val shape: Shape, val elements: DoubleArray) : Tensor<Double> {
    constructor(shape: Shape, element: Double = 0.0) : this(
        shape,
        doubleArrayOf(shape.volume.toDouble(), element)
    )

    // Companion object (similar to static context in Java)
    companion object {
        val doubleDataDescriptor = BuiltInDoubleDataDescriptor()

    }

    override val dataDescriptor: DataDescriptor
        get() = doubleDataDescriptor


    internal fun index(indices: IntArray): Int {
        assert(
            { indices.size == shape.dimensions.size },
            { "`indices.size` must be ${shape.dimensions.size}: ${indices.size}" })
        return shape.dimensions.zip(indices).fold(0) { a, x ->
            assert({ 0 <= x.second && x.second < x.first }, { "Illegal index: indices = ${indices}, shape = $shape" })
            a * x.first + x.second
        }
    }

    override operator fun get(vararg indices: Int): Double {
        return elements[index(indices)]
    }

    override operator fun get(vararg ranges: IntRange): Tensor<Double> {
        val size = ranges.size
        val shape = ranges.map { x -> x.last - x.first + 1 }
        val reversedShape = shape.reversed()
        val indices = IntArray(size)
        val elements = DoubleArray(shape.fold(1, Int::times)) {
            var i = it
            var dimensionIndex = size - 1
            for (dimension in reversedShape) {
                indices[dimensionIndex] = i % dimension + ranges[dimensionIndex].first
                i /= dimension
                dimensionIndex--
            }
            get(*indices)
        }
        return DoublesTensor(Shape(*shape), elements)
    }


    private inline fun commutativeBinaryOperation(
        tensor: DoublesTensor,
        operation: (Double, Double) -> Double
    ): Tensor<Double> {
        val lSize = shape.dimensions.size
        val rSize = tensor.shape.dimensions.size

        if (lSize == rSize) {
            assert(
                { shape == tensor.shape },
                { "Incompatible shapes of tensors: this.shape = ${shape}, tensor.shape = ${tensor.shape}" })
            return DoublesTensor(shape, zipMap(elements, tensor.elements, operation))
        }

        val a: DoublesTensor
        val b: DoublesTensor
        if (lSize < rSize) {
            a = tensor
            b = this
        } else {
            a = this
            b = tensor
        }
        assert(
            { a.shape.dimensions.endsWith(b.shape.dimensions) },
            { "Incompatible shapes of tensors: this.shape = ${shape}, tensor.shape = ${tensor.shape}" })

        return DoublesTensor(a.shape, zipMapRepeat(a.elements, b.elements, operation))
    }

    private inline fun noncommutativeBinaryOperation(
        tensor: DoublesTensor,
        operation: (Double, Double) -> Double,
        reverseOperation: (Double, Double) -> Double
    ): DoublesTensor {
        val lSize = shape.dimensions.size
        val rSize = tensor.shape.dimensions.size

        if (lSize == rSize) {
            assert(
                { shape == tensor.shape },
                { "Incompatible shapes of tensors: this.shape = ${shape}, tensor.shape = ${tensor.shape}" })
            return DoublesTensor(shape, zipMap(elements, tensor.elements, operation))
        } else if (lSize < rSize) {
            assert(
                { tensor.shape.dimensions.endsWith(shape.dimensions) },
                { "Incompatible shapes of tensors: this.shape = ${shape}, tensor.shape = ${tensor.shape}" })
            return DoublesTensor(tensor.shape, zipMapRepeat(tensor.elements, elements, reverseOperation))
        } else {
            assert(
                { shape.dimensions.endsWith(tensor.shape.dimensions) },
                { "Incompatible shapes of tensors: this.shape = ${shape}, tensor.shape = ${tensor.shape}" })
            return DoublesTensor(shape, zipMapRepeat(elements, tensor.elements, operation))
        }
    }

    operator fun plus(tensor: DoublesTensor): Tensor<Double> {
        return commutativeBinaryOperation(tensor) { lhs, rhs -> lhs + rhs }
    }

    operator fun plus(scalar: Double): Tensor<Double> {
        return DoublesTensor(shape, elements.map { it + scalar }.toDoubleArray())
    }

    operator fun minus(tensor: Tensor<Double>): Tensor<Double> {
        return noncommutativeBinaryOperation(
            tensor as DoublesTensor,
            { lhs, rhs -> lhs - rhs },
            { lhs, rhs -> rhs - lhs })
    }

    operator fun minus(scalar: Double): Tensor<Double> {
        return noncommutativeBinaryOperation(
            DoublesTensor(Shape(1), scalar),
            { lhs, rhs -> lhs - rhs },
            { lhs, rhs -> rhs - lhs })
    }


    operator fun times(tensor: Tensor<Double>): Tensor<Double> {
        return commutativeBinaryOperation(tensor as DoublesTensor) { lhs, rhs -> lhs * rhs }
    }

    operator fun div(tensor: Tensor<Double>): Tensor<Double> {
        return noncommutativeBinaryOperation(
            tensor as DoublesTensor,
            { lhs, rhs -> lhs / rhs },
            { lhs, rhs -> rhs / lhs })
    }

    operator fun times(scalar: Double): Tensor<Double> {
        return DoublesTensor(shape, elements.map { it * scalar }.toDoubleArray())
    }

    operator fun div(scalar: Double): Tensor<Double> {
        return DoublesTensor(shape, elements.map { it / scalar }.toDoubleArray())
    }

    fun matmul(other: Tensor<Double>): Tensor<Double> {
        // Scalar multiplication
        if (shape.dimensions.isEmpty() && other.shape.dimensions.isEmpty()) {
            return DoublesTensor(Shape(), doubleArrayOf(elements[0] * (other as DoublesTensor).elements[0]))
        }

        // Scalar and Vector multiplication (scalar is `this`)
        if (shape.dimensions.isEmpty()) {
            return DoublesTensor(
                other.shape,
                (other as DoublesTensor).elements.map { (it * elements[0]) }.toList().toDoubleArray()
            )
        }

        // Scalar and Vector multiplication (scalar is `other`)
        if (other.shape.dimensions.isEmpty()) {
            return DoublesTensor(
                shape,
                elements.map { it * (other as DoublesTensor).elements[0] }.toList().toDoubleArray()
            )
        }

        // Vector and Matrix multiplication
        if (shape.dimensions.size == 1 && other.shape.dimensions.size == 2) {
            if (shape.dimensions[0] != other.shape.dimensions[0]) throw IllegalArgumentException("Shapes do not align.")
            val result = DoubleArray(other.shape.dimensions[1]) { 0.0 }
            for (i in elements.indices) {
                for (j in 0 until other.shape.dimensions[1]) {
                    result[j] += elements[i] * (other as DoublesTensor).elements[i * other.shape.dimensions[1] + j]
                }
            }
            return DoublesTensor(Shape(other.shape.dimensions[1]), result)
        }

        // Matrix and Matrix multiplication
        if (shape.dimensions.size == 2 && other.shape.dimensions.size == 2) {
            if (shape.dimensions[1] != other.shape.dimensions[0]) throw IllegalArgumentException("Shapes do not align.")
            val newShape = Shape(shape.dimensions[0], other.shape.dimensions[1])
            val result = DoubleArray(newShape.volume) { 0.0 }
            for (i in 0 until shape.dimensions[0]) {
                for (j in 0 until other.shape.dimensions[1]) {
                    for (k in 0 until shape.dimensions[1]) {
                        result[i * newShape.dimensions[1] + j] += elements[i * shape.dimensions[1] + k] * (other as DoublesTensor).elements[k * other.shape.dimensions[1] + j]
                    }
                }
            }
            return DoublesTensor(newShape, result)
        }

        throw IllegalArgumentException("Unsupported tensor shapes for multiplication.")
    }

    override fun toString(): String {
        return when (shape.dimensions.size) {
            1 -> { // 1D tensor
                //println(shape)
                vectorToString()
            }

            2 -> { // 2D tensor
                //println(shape)
                matrixToString()
            }

            else -> "Tensor(${shape}, ${elements.contentToString()})" // higher dimensions
        }
    }

    private fun vectorToString(): String {
        return elements.joinToString(prefix = "[", postfix = "]")
    }

    private fun matrixToString(): String {
        val (rows, cols) = shape.dimensions
        return (0 until rows).joinToString(separator = "\n", prefix = "[\n", postfix = "\n]") { r ->
            (0 until cols).joinToString(prefix = " [", postfix = "]") { c ->
                elements[r * cols + c].toString()
            }
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as DoublesTensor

        if (shape != other.shape) return false
        if (!elements.contentEquals(other.elements)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = shape.hashCode()
        result = 31 * result + elements.contentHashCode()
        return result
    }
}

fun Tensor<Double>.matmul(other: Tensor<Double>): Tensor<Double> {
    return (this as DoublesTensor).matmul(other)
}

fun Tensor<Double>.elements() = (this as DoublesTensor).elements

operator fun Tensor<Double>.plus(other: Tensor<Double>): Tensor<Double> {
    return (this as DoublesTensor).matmul(other)
}
