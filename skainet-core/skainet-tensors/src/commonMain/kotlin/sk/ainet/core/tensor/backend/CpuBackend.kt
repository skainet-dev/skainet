package sk.ainet.core.tensor.backend

import sk.ainet.core.tensor.*
import sk.ainet.core.tensor.backend.ComputeBackend
import kotlin.math.*

/**
 * Convenient type alias for FP32 tensors with Float values.
 * Use this instead of concrete implementations where possible for better abstraction.
 */
public typealias TensorFP32 = Tensor<FP32, Float>

/**
 * @deprecated Use the generic tensor printing extensions from sk.ainet.core.tensor.printScalar() instead.
 * This CPU-specific function will be removed in a future version.
 */
@Deprecated(
    message = "Use the generic tensor printing extensions instead",
    replaceWith = ReplaceWith("this.printScalar()", "sk.ainet.core.tensor.printScalar"),
    level = DeprecationLevel.WARNING
)
public fun TensorFP32.printScalar(): String {
    require(shape.rank == 1 && shape[0] == 1) { "Tensor must be a scalar (1D with single element), got rank ${shape.rank} with shape ${shape.dimensions.contentToString()}" }
    return this[0].toString()
}

/**
 * @deprecated Use the generic tensor printing extensions from sk.ainet.core.tensor.printVector() instead.
 * This CPU-specific function will be removed in a future version.
 */
@Deprecated(
    message = "Use the generic tensor printing extensions instead",
    replaceWith = ReplaceWith("this.printVector()", "sk.ainet.core.tensor.printVector"),
    level = DeprecationLevel.WARNING
)
public fun TensorFP32.printVector(): String {
    require(shape.rank == 1) { "Tensor must be a vector (1D), got rank ${shape.rank}" }
    val elements = (0 until shape[0]).map { this[it] }
    return "[${elements.joinToString(", ")}]"
}

/**
 * @deprecated Use the generic tensor printing extensions from sk.ainet.core.tensor.printMatrix() instead.
 * This CPU-specific function will be removed in a future version.
 */
@Deprecated(
    message = "Use the generic tensor printing extensions instead",
    replaceWith = ReplaceWith("this.printMatrix()", "sk.ainet.core.tensor.printMatrix"),
    level = DeprecationLevel.WARNING
)
public fun TensorFP32.printMatrix(): String {
    require(shape.rank == 2) { "Tensor must be a matrix (2D), got rank ${shape.rank}" }
    val rows = shape[0]
    val cols = shape[1]
    val result = StringBuilder()

    result.append("[\n")
    for (i in 0 until rows) {
        result.append("  [")
        for (j in 0 until cols) {
            result.append(this[i, j])
            if (j < cols - 1) result.append(", ")
        }
        result.append("]")
        if (i < rows - 1) result.append(",")
        result.append("\n")
    }
    result.append("]")

    return result.toString()
}

/**
 * @deprecated Use the generic tensor printing extensions from sk.ainet.core.tensor.print() instead.
 * This CPU-specific function will be removed in a future version.
 */
@Deprecated(
    message = "Use the generic tensor printing extensions instead",
    replaceWith = ReplaceWith("this.print()", "sk.ainet.core.tensor.print"),
    level = DeprecationLevel.WARNING
)
public fun TensorFP32.print(): String {
    return when (shape.rank) {
        1 if shape[0] == 1 -> printScalar()
        1 -> printVector()
        2 -> printMatrix()
        else -> "Tensor(shape=${shape}, rank=${shape.rank}) - printing not supported for tensors with rank > 2"
    }
}


/**
 * A CPU-based tensor for FP32/Float values.
 *
 * This tensor stores data on the CPU using simple FloatArray with NCHW row-major layout.
 * It supports 1-4 dimensional tensors and delegates all operations to CpuBackend.
 */
public class CpuTensorFP32(
    override val shape: Shape,
    internal val data: FloatArray
) : TensorFP32 {

    init {
        require(data.size == shape.volume) {
            "Data size ${data.size} doesn't match shape volume ${shape.volume}"
        }
        require(shape.rank in 1..4) {
            "Only 1-4 dimensional tensors are supported, got ${shape.rank}"
        }
    }

    override fun get(vararg indices: Int): Float {
        val index = shape.index(indices)
        return data[index]
    }

    // Delegate all operations to the CpuBackend
    private val backend = CpuBackend()

    override fun matmul(a: Tensor<FP32, Float>, b: Tensor<FP32, Float>): Tensor<FP32, Float> = backend.matmul(a, b)
    override fun scale(a: Tensor<FP32, Float>, scalar: Double): Tensor<FP32, Float> = backend.scale(a, scalar)
    override fun dot(a: Tensor<FP32, Float>, b: Tensor<FP32, Float>): Double = backend.dot(a, b)

    // Tensor-Tensor operations
    override fun Tensor<FP32, Float>.plus(other: Tensor<FP32, Float>): Tensor<FP32, Float> =
        with(backend) { this@plus.plus(other) }

    override fun Tensor<FP32, Float>.minus(other: Tensor<FP32, Float>): Tensor<FP32, Float> =
        with(backend) { this@minus.minus(other) }

    override fun Tensor<FP32, Float>.times(other: Tensor<FP32, Float>): Tensor<FP32, Float> =
        with(backend) { this@times.times(other) }

    override fun Tensor<FP32, Float>.div(other: Tensor<FP32, Float>): Tensor<FP32, Float> =
        with(backend) { this@div.div(other) }

    // Tensor-Scalar operations - delegate to backend
    override fun Tensor<FP32, Float>.plus(scalar: Int): Tensor<FP32, Float> = with(backend) { this@plus.plus(scalar) }
    override fun Tensor<FP32, Float>.minus(scalar: Int): Tensor<FP32, Float> =
        with(backend) { this@minus.minus(scalar) }

    override fun Tensor<FP32, Float>.times(scalar: Int): Tensor<FP32, Float> =
        with(backend) { this@times.times(scalar) }

    override fun Tensor<FP32, Float>.div(scalar: Int): Tensor<FP32, Float> = with(backend) { this@div.div(scalar) }

    override fun Tensor<FP32, Float>.plus(scalar: Float): Tensor<FP32, Float> = with(backend) { this@plus.plus(scalar) }
    override fun Tensor<FP32, Float>.minus(scalar: Float): Tensor<FP32, Float> =
        with(backend) { this@minus.minus(scalar) }

    override fun Tensor<FP32, Float>.times(scalar: Float): Tensor<FP32, Float> =
        with(backend) { this@times.times(scalar) }

    override fun Tensor<FP32, Float>.div(scalar: Float): Tensor<FP32, Float> = with(backend) { this@div.div(scalar) }

    override fun Tensor<FP32, Float>.plus(scalar: Double): Tensor<FP32, Float> =
        with(backend) { this@plus.plus(scalar) }

    override fun Tensor<FP32, Float>.minus(scalar: Double): Tensor<FP32, Float> =
        with(backend) { this@minus.minus(scalar) }

    override fun Tensor<FP32, Float>.times(scalar: Double): Tensor<FP32, Float> =
        with(backend) { this@times.times(scalar) }

    override fun Tensor<FP32, Float>.div(scalar: Double): Tensor<FP32, Float> = with(backend) { this@div.div(scalar) }

    // Scalar-Tensor operations - delegate to backend
    override fun Double.plus(t: Tensor<FP32, Float>): Tensor<FP32, Float> = with(backend) { this@plus.plus(t) }
    override fun Double.minus(t: Tensor<FP32, Float>): Tensor<FP32, Float> = with(backend) { this@minus.minus(t) }
    override fun Double.times(t: Tensor<FP32, Float>): Tensor<FP32, Float> = with(backend) { this@times.times(t) }
    override fun Double.div(t: Tensor<FP32, Float>): Tensor<FP32, Float> = with(backend) { this@div.div(t) }

    // Advanced tensor operations - delegate to backend
    override fun Tensor<FP32, Float>.t(): Tensor<FP32, Float> = with(backend) { this@t.t() }
    override fun Tensor<FP32, Float>.relu(): Tensor<FP32, Float> = with(backend) { this@relu.relu() }
    override fun Tensor<FP32, Float>.sigmoid(): Tensor<FP32, Float> = with(backend) { this@sigmoid.sigmoid() }
    override fun Tensor<FP32, Float>.tanh(): Tensor<FP32, Float> = with(backend) { this@tanh.tanh() }
    override fun Tensor<FP32, Float>.softmax(dimension: Int): Tensor<FP32, Float> = with(backend) { this@softmax.softmax(dimension) }
    override fun Tensor<FP32, Float>.flatten(startDim: Int, endDim: Int): Tensor<FP32, Float> = with(backend) { this@flatten.flatten(startDim, endDim) }

    public companion object {
        /**
         * Creates a tensor from an array with the given shape.
         */
        public fun fromArray(shape: Shape, data: FloatArray): CpuTensorFP32 {
            return CpuTensorFP32(shape, data)
        }

        /**
         * Creates a tensor filled with zeros.
         */
        public fun zeros(shape: Shape): CpuTensorFP32 {
            return CpuTensorFP32(shape, FloatArray(shape.volume))
        }

        /**
         * Creates a tensor filled with ones.
         */
        public fun ones(shape: Shape): CpuTensorFP32 {
            return CpuTensorFP32(shape, FloatArray(shape.volume) { 1.0f })
        }

        /**
         * Creates a tensor filled with a specific value.
         */
        public fun full(shape: Shape, value: Float): CpuTensorFP32 {
            return CpuTensorFP32(shape, FloatArray(shape.volume) { value })
        }
    }
}

/**
 * A CPU-based implementation of the ComputeBackend interface for FP32/Float tensors.
 */
public class CpuBackend : ComputeBackend<FP32, Float> {
    override val name: String = "CPU"

    // Basic operations - implement the actual computation logic
    override fun matmul(a: Tensor<FP32, Float>, b: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(a is CpuTensorFP32 && b is CpuTensorFP32) { "Both tensors must be CpuTensorFP32" }
        require(a.shape.rank == 2 && b.shape.rank == 2) { "Matrix multiplication requires 2D tensors" }
        require(a.shape[1] == b.shape[0]) { "Matrix dimensions don't match for multiplication" }

        val rows = a.shape[0]
        val cols = b.shape[1]
        val inner = a.shape[1]
        val result = FloatArray(rows * cols)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                var sum = 0f
                for (k in 0 until inner) {
                    sum += a.data[i * inner + k] * b.data[k * cols + j]
                }
                result[i * cols + j] = sum
            }
        }

        return CpuTensorFP32(Shape(rows, cols), result)
    }

    override fun scale(a: Tensor<FP32, Float>, scalar: Double): Tensor<FP32, Float> {
        require(a is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = a.data.map { it * scalar.toFloat() }.toFloatArray()
        return CpuTensorFP32(a.shape, result)
    }

    override fun dot(a: Tensor<FP32, Float>, b: Tensor<FP32, Float>): Double {
        require(a is CpuTensorFP32 && b is CpuTensorFP32) { "Both tensors must be CpuTensorFP32" }
        require(a.shape == b.shape) { "Tensors must have same shape for dot product" }

        var sum = 0.0
        for (i in a.data.indices) {
            sum += a.data[i] * b.data[i]
        }
        return sum
    }

    // Tensor-Tensor operations - implement actual computation logic
    override fun Tensor<FP32, Float>.plus(other: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(this is CpuTensorFP32 && other is CpuTensorFP32) { "Both tensors must be CpuTensorFP32" }
        require(this.shape == other.shape) { "Tensors must have same shape for addition" }

        val result = FloatArray(this.data.size)
        for (i in this.data.indices) {
            result[i] = this.data[i] + other.data[i]
        }
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.minus(other: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(this is CpuTensorFP32 && other is CpuTensorFP32) { "Both tensors must be CpuTensorFP32" }
        require(this.shape == other.shape) { "Tensors must have same shape for subtraction" }

        val result = FloatArray(this.data.size)
        for (i in this.data.indices) {
            result[i] = this.data[i] - other.data[i]
        }
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.times(other: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(this is CpuTensorFP32 && other is CpuTensorFP32) { "Both tensors must be CpuTensorFP32" }
        require(this.shape == other.shape) { "Tensors must have same shape for element-wise multiplication" }

        val result = FloatArray(this.data.size)
        for (i in this.data.indices) {
            result[i] = this.data[i] * other.data[i]
        }
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.div(other: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(this is CpuTensorFP32 && other is CpuTensorFP32) { "Both tensors must be CpuTensorFP32" }
        require(this.shape == other.shape) { "Tensors must have same shape for element-wise division" }

        val result = FloatArray(this.data.size)
        for (i in this.data.indices) {
            result[i] = this.data[i] / other.data[i]
        }
        return CpuTensorFP32(this.shape, result)
    }

    // Tensor-Scalar operations
    override fun Tensor<FP32, Float>.plus(scalar: Int): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { it + scalar }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.minus(scalar: Int): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { it - scalar }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.times(scalar: Int): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { it * scalar }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.div(scalar: Int): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { it / scalar }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.plus(scalar: Float): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { it + scalar }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.minus(scalar: Float): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { it - scalar }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.times(scalar: Float): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { it * scalar }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.div(scalar: Float): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { it / scalar }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.plus(scalar: Double): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { it + scalar.toFloat() }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.minus(scalar: Double): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { it - scalar.toFloat() }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.times(scalar: Double): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { it * scalar.toFloat() }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.div(scalar: Double): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { it / scalar.toFloat() }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }

    // Scalar-Tensor operations
    override fun Double.plus(t: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(t is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = t.data.map { this.toFloat() + it }.toFloatArray()
        return CpuTensorFP32(t.shape, result)
    }

    override fun Double.minus(t: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(t is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = t.data.map { this.toFloat() - it }.toFloatArray()
        return CpuTensorFP32(t.shape, result)
    }

    override fun Double.times(t: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(t is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = t.data.map { this.toFloat() * it }.toFloatArray()
        return CpuTensorFP32(t.shape, result)
    }

    override fun Double.div(t: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(t is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = t.data.map { this.toFloat() / it }.toFloatArray()
        return CpuTensorFP32(t.shape, result)
    }

    // Advanced tensor operations
    override fun Tensor<FP32, Float>.t(): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        require(this.shape.rank == 2) { "Transpose only supported for 2D tensors (matrices)" }
        
        val rows = this.shape[0]
        val cols = this.shape[1]
        val result = FloatArray(rows * cols)
        
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result[j * rows + i] = this.data[i * cols + j]
            }
        }
        
        return CpuTensorFP32(Shape(cols, rows), result)
    }

    override fun Tensor<FP32, Float>.relu(): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { maxOf(0f, it) }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.sigmoid(): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { 1f / (1f + exp(-it)) }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.tanh(): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { tanh(it) }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.softmax(dimension: Int): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        require(dimension in 0 until this.shape.rank) { "Dimension $dimension is out of bounds for tensor with rank ${this.shape.rank}" }
        
        when (this.shape.rank) {
            1 -> {
                // For 1D tensor, apply softmax across the single dimension
                val maxVal = this.data.maxOrNull() ?: 0f
                val expValues = this.data.map { exp(it - maxVal) }
                val sum = expValues.sum()
                val result = expValues.map { it / sum }.toFloatArray()
                return CpuTensorFP32(this.shape, result)
            }
            2 -> {
                // For 2D tensor (matrix), apply softmax along specified dimension
                val rows = this.shape[0]
                val cols = this.shape[1]
                val result = FloatArray(this.data.size)
                
                if (dimension == 0) {
                    // Apply softmax along rows (for each column)
                    for (j in 0 until cols) {
                        val columnValues = FloatArray(rows) { i -> this.data[i * cols + j] }
                        val maxVal = columnValues.maxOrNull() ?: 0f
                        val expValues = columnValues.map { exp(it - maxVal) }
                        val sum = expValues.sum()
                        for (i in 0 until rows) {
                            result[i * cols + j] = expValues[i] / sum
                        }
                    }
                } else {
                    // Apply softmax along columns (for each row)
                    for (i in 0 until rows) {
                        val rowStart = i * cols
                        val rowValues = this.data.sliceArray(rowStart until rowStart + cols)
                        val maxVal = rowValues.maxOrNull() ?: 0f
                        val expValues = rowValues.map { exp(it - maxVal) }
                        val sum = expValues.sum()
                        for (j in 0 until cols) {
                            result[rowStart + j] = expValues[j] / sum
                        }
                    }
                }
                return CpuTensorFP32(this.shape, result)
            }
            else -> {
                throw UnsupportedOperationException("Softmax not implemented for tensors with rank > 2")
            }
        }
    }

    override fun Tensor<FP32, Float>.flatten(startDim: Int, endDim: Int): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        
        val actualEndDim = if (endDim == -1) this.shape.rank - 1 else endDim
        require(startDim >= 0 && startDim < this.shape.rank) { "startDim $startDim is out of bounds" }
        require(actualEndDim >= startDim && actualEndDim < this.shape.rank) { "endDim $actualEndDim is out of bounds or less than startDim" }
        
        if (startDim == actualEndDim) {
            // No flattening needed
            return CpuTensorFP32(this.shape, this.data.copyOf())
        }
        
        // Calculate new shape
        val newDimensions = mutableListOf<Int>()
        
        // Add dimensions before startDim
        for (i in 0 until startDim) {
            newDimensions.add(this.shape[i])
        }
        
        // Calculate flattened dimension size
        var flattenedSize = 1
        for (i in startDim..actualEndDim) {
            flattenedSize *= this.shape[i]
        }
        newDimensions.add(flattenedSize)
        
        // Add dimensions after endDim
        for (i in (actualEndDim + 1) until this.shape.rank) {
            newDimensions.add(this.shape[i])
        }
        
        val newShape = Shape(*newDimensions.toIntArray())
        return CpuTensorFP32(newShape, this.data.copyOf())
    }
}