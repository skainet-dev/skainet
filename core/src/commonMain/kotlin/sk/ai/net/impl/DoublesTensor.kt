package sk.ai.net.impl

import sk.ai.net.DataDescriptor
import sk.ai.net.Shape
import sk.ai.net.Tensor
import kotlin.collections.map
import kotlin.math.exp
import kotlin.math.pow

data class DoublesTensor(override val shape: Shape, val elements: DoubleArray) : TypedTensor<Double> {
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

    override operator fun get(vararg ranges: IntRange): TypedTensor<Double> {
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
        return DoublesTensor(Shape(*shape.toIntArray()), elements)
    }


    private inline fun commutativeBinaryOperation(
        tensor: DoublesTensor,
        operation: (Double, Double) -> Double
    ): TypedTensor<Double> {
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

    private inline fun nonCommutativeBinaryOperation(
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

    override operator fun plus(other: Tensor): Tensor {
        return commutativeBinaryOperation(other as DoublesTensor) { lhs, rhs -> lhs + rhs }
    }

    override operator fun plus(other: Double): Tensor {
        return commutativeBinaryOperation(
            DoublesTensor(Shape(1), other)
        ) { lhs, rhs -> lhs + rhs }
    }

    override fun plus(other: Int): Tensor {
        return commutativeBinaryOperation(
            DoublesTensor(Shape(1), other.toDouble())
        ) { lhs, rhs -> lhs + rhs }
    }

    override operator fun minus(other: Tensor): Tensor {
        return nonCommutativeBinaryOperation(
            other as DoublesTensor,
            { lhs, rhs -> lhs - rhs },
            { lhs, rhs -> rhs - lhs })
    }

    override operator fun minus(other: Double): TypedTensor<Double> {
        return nonCommutativeBinaryOperation(
            DoublesTensor(Shape(1), other),
            { lhs, rhs -> lhs - rhs },
            { lhs, rhs -> rhs - lhs })
    }

    override fun minus(other: Int): Tensor {
        return nonCommutativeBinaryOperation(
            DoublesTensor(Shape(1), other.toDouble()),
            { lhs, rhs -> lhs - rhs },
            { lhs, rhs -> rhs - lhs })
    }


    operator fun times(tensor: TypedTensor<Double>): TypedTensor<Double> {
        return commutativeBinaryOperation(tensor as DoublesTensor) { lhs, rhs -> lhs * rhs }
    }

    operator fun div(tensor: TypedTensor<Double>): TypedTensor<Double> {
        return nonCommutativeBinaryOperation(
            tensor as DoublesTensor,
            { lhs, rhs -> lhs / rhs },
            { lhs, rhs -> rhs / lhs })
    }

    operator fun times(scalar: Double): TypedTensor<Double> {
        return DoublesTensor(shape, elements.map { it * scalar }.toDoubleArray())
    }

    operator fun div(scalar: Double): TypedTensor<Double> {
        return DoublesTensor(shape, elements.map { it / scalar }.toDoubleArray())
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


    override fun matmul(other: Tensor): Tensor {
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

    override fun t(): Tensor {
        // Ensure the tensor is 2D
        if (this.shape.dimensions.size != 2) {
            throw IllegalArgumentException("Transpose is only implemented for 2D tensors.")
        }

        // New shape with dimensions swapped
        val newShape = Shape(this.shape.dimensions[1], this.shape.dimensions[0])

        // Create a new elements array to hold the transposed elements
        val newElements = DoubleArray(this.elements.size)

        // Populate the new elements array with the transposed elements
        for (i in 0 until shape.dimensions[0]) { // Original rows
            for (j in 0 until shape.dimensions[1]) { // Original columns
                // Calculate the index in the original flat array and the new index in the transposed array
                val originalIndex = i * shape.dimensions[1] + j
                val newIndex = j * shape.dimensions[0] + i
                // Assign the transposed value
                newElements[newIndex] = this.elements[originalIndex]
            }
        }

        // Return a new tensor with the transposed shape and elements
        return DoublesTensor(newShape, newElements)
    }

    override fun relu(): Tensor =
        DoublesTensor(shape, elements.map { elem -> if (elem > 0) elem else 0.0 }.toDoubleArray())

    override fun softmax(): Tensor {
        val sum = elements.fold(0.0) { r, x -> r + x }
        return this / sum
    }

    override fun pow(tensor: Tensor): Tensor {
        assert(
            { shape == tensor.shape },
            { "Incompatible shapes of tensors: this.shape = ${shape}, tensor.shape = ${tensor.shape}" })
        return DoublesTensor(shape, zipMap(elements, (tensor as DoublesTensor).elements) { a, b -> a.pow(b) })
    }

    override fun pow(scalar: Double): Tensor {
        TODO("Not yet implemented")
    }


    override fun sin(): Tensor =
        DoublesTensor(shape, elements.map { kotlin.math.sin(it) }.toDoubleArray())

    override fun cos(): Tensor =
        DoublesTensor(shape, elements.map { kotlin.math.cos(it) }.toDoubleArray())

    override fun tan(): Tensor =
        DoublesTensor(shape, elements.map { kotlin.math.tan(it) }.toDoubleArray())

    override fun asin(): Tensor =
        DoublesTensor(shape, elements.map { kotlin.math.asin(it) }.toDoubleArray())

    override fun acos(): Tensor =
        DoublesTensor(shape, elements.map { kotlin.math.acos(it) }.toDoubleArray())

    override fun atan(): Tensor =
        DoublesTensor(shape, elements.map { kotlin.math.atan(it) }.toDoubleArray())

    override fun sinh(): Tensor =
        DoublesTensor(shape, elements.map { kotlin.math.sinh(it) }.toDoubleArray())

    override fun cosh(): Tensor =
        DoublesTensor(shape, elements.map { kotlin.math.cosh(it) }.toDoubleArray())

    override fun tanh(): Tensor =
        DoublesTensor(shape, elements.map { kotlin.math.tanh(it) }.toDoubleArray())

    override fun exp(): Tensor =
        DoublesTensor(shape, elements.map { exp(it) }.toDoubleArray())

    override fun log(): Tensor =
        DoublesTensor(shape, elements.map { kotlin.math.ln(it) }.toDoubleArray())

    override fun sqrt(): Tensor =
        DoublesTensor(shape, elements.map { kotlin.math.sqrt(it) }.toDoubleArray())

    override fun cbrt(): Tensor =
        DoublesTensor(shape, elements.map { kotlin.math.cbrt(it) }.toDoubleArray())

    override fun sigmoid(): Tensor =
        DoublesTensor(shape, elements.map { (1.0 / exp(-it)) }.toDoubleArray())


    override fun ln(): Tensor =
        DoublesTensor(shape, elements.map { kotlin.math.ln(it) }.toDoubleArray())

    fun computeStrides(dimensions: IntArray): IntArray {
        val strides = IntArray(dimensions.size) { 1 }
        for (i in dimensions.lastIndex - 1 downTo 0) {
            strides[i] = strides[i + 1] * dimensions[i + 1]
        }
        return strides
    }

    fun unravelIndex(index: Int, dimensions: IntArray, strides: IntArray): IntArray {
        var idx = index
        val indices = IntArray(dimensions.size)
        for (i in strides.indices) {
            indices[i] = idx / strides[i]
            idx %= strides[i]
        }
        return indices
    }


    override fun softmax(dim: Int): Tensor {
        val actualDim = if (dim < 0) shape.dimensions.size + dim else dim
        if (actualDim < 0 || actualDim >= shape.dimensions.size) {
            throw IllegalArgumentException("Dimension out of range")
        }

        // Compute the exponential of each element and the sum of exponential along the specified dimension.
        val exps = DoubleArray(elements.size)
        val sumExps = DoubleArray(shape.volume / shape.dimensions[actualDim]) { 0.0 }

        val strides = computeStrides(shape.dimensions)
        for (index in elements.indices) {
            val indices = unravelIndex(index, shape.dimensions, strides)
            val dimIndex = indices[actualDim]
            val exp = exp(elements[index])
            exps[index] = exp
            sumExps[dimIndex] += exp
        }

        // Normalize by the sum of exponential to get softmax probabilities.
        val softmaxElements = DoubleArray(elements.size)
        for (index in elements.indices) {
            val indices = unravelIndex(index, shape.dimensions, strides)
            val dimIndex = indices[actualDim]
            softmaxElements[index] = exps[index] / sumExps[dimIndex]
        }

        return DoublesTensor(shape, softmaxElements)
    }
}

fun DoublesTensor.prod(): Double = this.elements.fold(1.0) { acc, element -> acc * element }
