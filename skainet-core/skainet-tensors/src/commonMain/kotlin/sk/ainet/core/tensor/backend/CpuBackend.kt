package sk.ainet.core.tensor.backend

import sk.ainet.core.tensor.*
import kotlin.math.*

/**
 * Object to initialize default tensor factories.
 * The init block ensures factories are set up when this class is first accessed.
 */
internal object TensorFactoryInitializer {
    init {
        DefaultTensorFactories.setFP32Factory(CpuBackend())
        DefaultTensorFactories.setInt8Factory(CpuBackendInt8())  
        DefaultTensorFactories.setInt32Factory(CpuBackendInt32())
    }
    
    /**
     * This function ensures the initializer is called.
     * Called by backend classes to guarantee initialization.
     */
    fun ensureInitialized() {
        // Just accessing this object ensures the init block runs
    }
}

/**
 * Convenient type alias for FP32 tensors with Float values.
 * Use this instead of concrete implementations where possible for better abstraction.
 */
public typealias TensorFP32 = Tensor<FP32, Float>

/**
 * Convenient type alias for Int8 tensors with Byte values.
 * Use this instead of concrete implementations where possible for better abstraction.
 */
public typealias TensorInt8 = Tensor<Int8, Byte>

/**
 * Convenient type alias for Int32 tensors with Int values.
 * Use this instead of concrete implementations where possible for better abstraction.
 */
public typealias TensorInt32 = Tensor<Int32, Int>

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
 * A CPU-based tensor for Int8/Byte values.
 *
 * This tensor stores data on the CPU using simple ByteArray with NCHW row-major layout.
 * It supports 1-4 dimensional tensors and implements operations directly.
 */
public class CpuTensorInt8(
    override val shape: Shape,
    internal val data: ByteArray
) : TensorInt8 {

    init {
        require(data.size == shape.volume) {
            "Data size ${data.size} doesn't match shape volume ${shape.volume}"
        }
        require(shape.rank in 1..4) {
            "Only 1-4 dimensional tensors are supported, got ${shape.rank}"
        }
    }

    override fun get(vararg indices: Int): Byte {
        val index = shape.index(indices)
        return data[index]
    }

    // Helper function to clamp values to Byte range
    private fun clampToByte(value: Double): Byte = value.toInt().coerceIn(Byte.MIN_VALUE.toInt(), Byte.MAX_VALUE.toInt()).toByte()

    // Basic operations - implement directly
    override fun matmul(a: Tensor<Int8, Byte>, b: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        require(a is CpuTensorInt8 && b is CpuTensorInt8) { "Both tensors must be CpuTensorInt8" }
        require(a.shape.rank == 2 && b.shape.rank == 2) { "Matrix multiplication requires 2D tensors" }
        require(a.shape[1] == b.shape[0]) { "Matrix dimensions don't match for multiplication" }

        val rows = a.shape[0]
        val cols = b.shape[1]
        val inner = a.shape[1]
        val result = ByteArray(rows * cols)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                var sum = 0
                for (k in 0 until inner) {
                    sum += a.data[i * inner + k] * b.data[k * cols + j]
                }
                result[i * cols + j] = clampToByte(sum.toDouble())
            }
        }

        return CpuTensorInt8(Shape(rows, cols), result)
    }

    override fun scale(a: Tensor<Int8, Byte>, scalar: Double): Tensor<Int8, Byte> {
        require(a is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val result = a.data.map { clampToByte(it * scalar) }.toByteArray()
        return CpuTensorInt8(a.shape, result)
    }

    override fun dot(a: Tensor<Int8, Byte>, b: Tensor<Int8, Byte>): Double {
        require(a is CpuTensorInt8 && b is CpuTensorInt8) { "Both tensors must be CpuTensorInt8" }
        require(a.shape == b.shape) { "Tensors must have same shape for dot product" }

        var sum = 0.0
        for (i in a.data.indices) {
            sum += a.data[i] * b.data[i]
        }
        return sum
    }

    // Tensor-Tensor operations
    override fun Tensor<Int8, Byte>.plus(other: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8 && other is CpuTensorInt8) { "Both tensors must be CpuTensorInt8" }
        require(canBroadcast(this.shape, other.shape)) { "Tensors are not broadcast-compatible for addition: ${this.shape} vs ${other.shape}" }

        val resultShape = getBroadcastShape(this.shape, other.shape)
        val result = ByteArray(resultShape.volume)
        
        for (i in result.indices) {
            val thisIndex = broadcastIndex(i, resultShape, this.shape)
            val otherIndex = broadcastIndex(i, resultShape, other.shape)
            result[i] = clampToByte((this.data[thisIndex] + other.data[otherIndex]).toDouble())
        }
        return CpuTensorInt8(resultShape, result)
    }

    override fun Tensor<Int8, Byte>.minus(other: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8 && other is CpuTensorInt8) { "Both tensors must be CpuTensorInt8" }
        require(canBroadcast(this.shape, other.shape)) { "Tensors are not broadcast-compatible for subtraction: ${this.shape} vs ${other.shape}" }

        val resultShape = getBroadcastShape(this.shape, other.shape)
        val result = ByteArray(resultShape.volume)
        
        for (i in result.indices) {
            val thisIndex = broadcastIndex(i, resultShape, this.shape)
            val otherIndex = broadcastIndex(i, resultShape, other.shape)
            result[i] = clampToByte((this.data[thisIndex] - other.data[otherIndex]).toDouble())
        }
        return CpuTensorInt8(resultShape, result)
    }

    override fun Tensor<Int8, Byte>.times(other: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8 && other is CpuTensorInt8) { "Both tensors must be CpuTensorInt8" }
        require(canBroadcast(this.shape, other.shape)) { "Tensors are not broadcast-compatible for element-wise multiplication: ${this.shape} vs ${other.shape}" }

        val resultShape = getBroadcastShape(this.shape, other.shape)
        val result = ByteArray(resultShape.volume)
        
        for (i in result.indices) {
            val thisIndex = broadcastIndex(i, resultShape, this.shape)
            val otherIndex = broadcastIndex(i, resultShape, other.shape)
            result[i] = clampToByte((this.data[thisIndex] * other.data[otherIndex]).toDouble())
        }
        return CpuTensorInt8(resultShape, result)
    }

    override fun Tensor<Int8, Byte>.div(other: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8 && other is CpuTensorInt8) { "Both tensors must be CpuTensorInt8" }
        require(canBroadcast(this.shape, other.shape)) { "Tensors are not broadcast-compatible for element-wise division: ${this.shape} vs ${other.shape}" }

        val resultShape = getBroadcastShape(this.shape, other.shape)
        val result = ByteArray(resultShape.volume)
        
        for (i in result.indices) {
            val thisIndex = broadcastIndex(i, resultShape, this.shape)
            val otherIndex = broadcastIndex(i, resultShape, other.shape)
            result[i] = if (other.data[otherIndex] != 0.toByte()) clampToByte((this.data[thisIndex] / other.data[otherIndex]).toDouble()) else 0
        }
        return CpuTensorInt8(resultShape, result)
    }

    // Tensor-Scalar operations
    override fun Tensor<Int8, Byte>.plus(scalar: Int): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val result = this.data.map { clampToByte((it + scalar).toDouble()) }.toByteArray()
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.minus(scalar: Int): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val result = this.data.map { clampToByte((it - scalar).toDouble()) }.toByteArray()
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.times(scalar: Int): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val result = this.data.map { clampToByte((it * scalar).toDouble()) }.toByteArray()
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.div(scalar: Int): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val result = this.data.map { if (scalar != 0) clampToByte((it / scalar).toDouble()) else 0 }.toByteArray()
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.plus(scalar: Float): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val result = this.data.map { clampToByte((it + scalar).toDouble()) }.toByteArray()
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.minus(scalar: Float): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val result = this.data.map { clampToByte((it - scalar).toDouble()) }.toByteArray()
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.times(scalar: Float): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val result = this.data.map { clampToByte((it * scalar).toDouble()) }.toByteArray()
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.div(scalar: Float): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val result = this.data.map { if (scalar != 0f) clampToByte((it / scalar).toDouble()) else 0 }.toByteArray()
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.plus(scalar: Double): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val result = this.data.map { clampToByte(it + scalar) }.toByteArray()
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.minus(scalar: Double): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val result = this.data.map { clampToByte(it - scalar) }.toByteArray()
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.times(scalar: Double): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val result = this.data.map { clampToByte(it * scalar) }.toByteArray()
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.div(scalar: Double): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val result = this.data.map { if (scalar != 0.0) clampToByte(it / scalar) else 0 }.toByteArray()
        return CpuTensorInt8(this.shape, result)
    }

    // Scalar-Tensor operations
    override fun Double.plus(t: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        require(t is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val result = t.data.map { clampToByte(this + it) }.toByteArray()
        return CpuTensorInt8(t.shape, result)
    }

    override fun Double.minus(t: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        require(t is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val result = t.data.map { clampToByte(this - it) }.toByteArray()
        return CpuTensorInt8(t.shape, result)
    }

    override fun Double.times(t: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        require(t is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val result = t.data.map { clampToByte(this * it) }.toByteArray()
        return CpuTensorInt8(t.shape, result)
    }

    override fun Double.div(t: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        require(t is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val result = t.data.map { if (it != 0.toByte()) clampToByte(this / it) else 0 }.toByteArray()
        return CpuTensorInt8(t.shape, result)
    }

    // Advanced tensor operations
    override fun Tensor<Int8, Byte>.t(): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        require(this.shape.rank == 2) { "Transpose only supported for 2D tensors (matrices)" }
        
        val rows = this.shape[0]
        val cols = this.shape[1]
        val result = ByteArray(rows * cols)
        
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result[j * rows + i] = this.data[i * cols + j]
            }
        }
        
        return CpuTensorInt8(Shape(cols, rows), result)
    }

    override fun Tensor<Int8, Byte>.relu(): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val result = this.data.map { maxOf(0, it.toInt()).toByte() }.toByteArray()
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.sigmoid(): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        // For integer tensors, sigmoid is approximated and scaled to byte range
        val result = this.data.map { 
            val sigmoid = 1.0 / (1.0 + exp(-it.toDouble() / 127.0))  // Scale input to [-1, 1] range
            clampToByte((sigmoid * 254 - 127).toDouble())  // Scale output to byte range
        }.toByteArray()
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.tanh(): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        // For integer tensors, tanh is approximated and scaled to byte range
        val result = this.data.map { 
            val tanhValue = tanh(it.toDouble() / 127.0)  // Scale input to [-1, 1] range
            clampToByte((tanhValue * 127).toDouble())  // Scale output to byte range
        }.toByteArray()
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.softmax(dimension: Int): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        require(dimension in 0 until this.shape.rank) { "Dimension $dimension is out of bounds for tensor with rank ${this.shape.rank}" }
        
        when (this.shape.rank) {
            1 -> {
                // Convert to float for softmax computation, then scale back
                val floatData = this.data.map { it.toFloat() / 127f }  // Scale to [-1, 1]
                val maxVal = floatData.maxOrNull() ?: 0f
                val expValues = floatData.map { exp(it - maxVal) }
                val sum = expValues.sum()
                val result = expValues.map { clampToByte(((it / sum) * 254 - 127).toDouble()) }.toByteArray()
                return CpuTensorInt8(this.shape, result)
            }
            2 -> {
                val rows = this.shape[0]
                val cols = this.shape[1]
                val result = ByteArray(this.data.size)
                
                if (dimension == 0) {
                    // Apply softmax along rows (for each column)
                    for (j in 0 until cols) {
                        val columnValues = FloatArray(rows) { i -> this.data[i * cols + j].toFloat() / 127f }
                        val maxVal = columnValues.maxOrNull() ?: 0f
                        val expValues = columnValues.map { exp(it - maxVal) }
                        val sum = expValues.sum()
                        for (i in 0 until rows) {
                            result[i * cols + j] = clampToByte(((expValues[i] / sum) * 254 - 127).toDouble())
                        }
                    }
                } else {
                    // Apply softmax along columns (for each row)
                    for (i in 0 until rows) {
                        val rowStart = i * cols
                        val rowValues = ByteArray(cols) { j -> this.data[rowStart + j] }.map { it.toFloat() / 127f }
                        val maxVal = rowValues.maxOrNull() ?: 0f
                        val expValues = rowValues.map { exp(it - maxVal) }
                        val sum = expValues.sum()
                        for (j in 0 until cols) {
                            result[rowStart + j] = clampToByte(((expValues[j] / sum) * 254 - 127).toDouble())
                        }
                    }
                }
                return CpuTensorInt8(this.shape, result)
            }
            else -> {
                throw UnsupportedOperationException("Softmax not implemented for tensors with rank > 2")
            }
        }
    }

    override fun Tensor<Int8, Byte>.flatten(startDim: Int, endDim: Int): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        
        val actualEndDim = if (endDim == -1) this.shape.rank - 1 else endDim
        require(startDim >= 0 && startDim < this.shape.rank) { "startDim $startDim is out of bounds" }
        require(actualEndDim >= startDim && actualEndDim < this.shape.rank) { "endDim $actualEndDim is out of bounds or less than startDim" }
        
        if (startDim == actualEndDim) {
            return CpuTensorInt8(this.shape, this.data.copyOf())
        }
        
        val newDimensions = mutableListOf<Int>()
        
        for (i in 0 until startDim) {
            newDimensions.add(this.shape[i])
        }
        
        var flattenedSize = 1
        for (i in startDim..actualEndDim) {
            flattenedSize *= this.shape[i]
        }
        newDimensions.add(flattenedSize)
        
        for (i in (actualEndDim + 1) until this.shape.rank) {
            newDimensions.add(this.shape[i])
        }
        
        val newShape = Shape(*newDimensions.toIntArray())
        return CpuTensorInt8(newShape, this.data.copyOf())
    }

    public companion object {
        /**
         * Creates a tensor from an array with the given shape.
         */
        public fun fromArray(shape: Shape, data: ByteArray): CpuTensorInt8 {
            return CpuTensorInt8(shape, data)
        }

        /**
         * Creates a tensor filled with zeros.
         */
        public fun zeros(shape: Shape): CpuTensorInt8 {
            return CpuTensorInt8(shape, ByteArray(shape.volume))
        }

        /**
         * Creates a tensor filled with ones.
         */
        public fun ones(shape: Shape): CpuTensorInt8 {
            return CpuTensorInt8(shape, ByteArray(shape.volume) { 1 })
        }

        /**
         * Creates a tensor filled with a specific value.
         */
        public fun full(shape: Shape, value: Byte): CpuTensorInt8 {
            return CpuTensorInt8(shape, ByteArray(shape.volume) { value })
        }
    }
}

/**
 * A CPU-based tensor for Int32/Int values.
 *
 * This tensor stores data on the CPU using simple IntArray with NCHW row-major layout.
 * It supports 1-4 dimensional tensors and implements operations directly.
 */
public class CpuTensorInt32(
    override val shape: Shape,
    internal val data: IntArray
) : TensorInt32 {

    init {
        require(data.size == shape.volume) {
            "Data size ${data.size} doesn't match shape volume ${shape.volume}"
        }
        require(shape.rank in 1..4) {
            "Only 1-4 dimensional tensors are supported, got ${shape.rank}"
        }
    }

    override fun get(vararg indices: Int): Int {
        val index = shape.index(indices)
        return data[index]
    }

    // Helper function to clamp values to Int range (overflow protection)
    private fun clampToInt(value: Long): Int = value.coerceIn(Int.MIN_VALUE.toLong(), Int.MAX_VALUE.toLong()).toInt()

    // Basic operations - implement directly
    override fun matmul(a: Tensor<Int32, Int>, b: Tensor<Int32, Int>): Tensor<Int32, Int> {
        require(a is CpuTensorInt32 && b is CpuTensorInt32) { "Both tensors must be CpuTensorInt32" }
        require(a.shape.rank == 2 && b.shape.rank == 2) { "Matrix multiplication requires 2D tensors" }
        require(a.shape[1] == b.shape[0]) { "Matrix dimensions don't match for multiplication" }

        val rows = a.shape[0]
        val cols = b.shape[1]
        val inner = a.shape[1]
        val result = IntArray(rows * cols)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                var sum = 0L
                for (k in 0 until inner) {
                    sum += a.data[i * inner + k].toLong() * b.data[k * cols + j].toLong()
                }
                result[i * cols + j] = clampToInt(sum)
            }
        }

        return CpuTensorInt32(Shape(rows, cols), result)
    }

    override fun scale(a: Tensor<Int32, Int>, scalar: Double): Tensor<Int32, Int> {
        require(a is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val result = a.data.map { clampToInt((it.toDouble() * scalar).toLong()) }.toIntArray()
        return CpuTensorInt32(a.shape, result)
    }

    override fun dot(a: Tensor<Int32, Int>, b: Tensor<Int32, Int>): Double {
        require(a is CpuTensorInt32 && b is CpuTensorInt32) { "Both tensors must be CpuTensorInt32" }
        require(a.shape == b.shape) { "Tensors must have same shape for dot product" }

        var sum = 0.0
        for (i in a.data.indices) {
            sum += a.data[i].toDouble() * b.data[i].toDouble()
        }
        return sum
    }

    // Tensor-Tensor operations
    override fun Tensor<Int32, Int>.plus(other: Tensor<Int32, Int>): Tensor<Int32, Int> {
        require(this is CpuTensorInt32 && other is CpuTensorInt32) { "Both tensors must be CpuTensorInt32" }
        require(canBroadcast(this.shape, other.shape)) { "Tensors are not broadcast-compatible for addition: ${this.shape} vs ${other.shape}" }

        val resultShape = getBroadcastShape(this.shape, other.shape)
        val result = IntArray(resultShape.volume)
        
        for (i in result.indices) {
            val thisIndex = broadcastIndex(i, resultShape, this.shape)
            val otherIndex = broadcastIndex(i, resultShape, other.shape)
            result[i] = clampToInt(this.data[thisIndex].toLong() + other.data[otherIndex].toLong())
        }
        return CpuTensorInt32(resultShape, result)
    }

    override fun Tensor<Int32, Int>.minus(other: Tensor<Int32, Int>): Tensor<Int32, Int> {
        require(this is CpuTensorInt32 && other is CpuTensorInt32) { "Both tensors must be CpuTensorInt32" }
        require(canBroadcast(this.shape, other.shape)) { "Tensors are not broadcast-compatible for subtraction: ${this.shape} vs ${other.shape}" }

        val resultShape = getBroadcastShape(this.shape, other.shape)
        val result = IntArray(resultShape.volume)
        
        for (i in result.indices) {
            val thisIndex = broadcastIndex(i, resultShape, this.shape)
            val otherIndex = broadcastIndex(i, resultShape, other.shape)
            result[i] = clampToInt(this.data[thisIndex].toLong() - other.data[otherIndex].toLong())
        }
        return CpuTensorInt32(resultShape, result)
    }

    override fun Tensor<Int32, Int>.times(other: Tensor<Int32, Int>): Tensor<Int32, Int> {
        require(this is CpuTensorInt32 && other is CpuTensorInt32) { "Both tensors must be CpuTensorInt32" }
        require(canBroadcast(this.shape, other.shape)) { "Tensors are not broadcast-compatible for element-wise multiplication: ${this.shape} vs ${other.shape}" }

        val resultShape = getBroadcastShape(this.shape, other.shape)
        val result = IntArray(resultShape.volume)
        
        for (i in result.indices) {
            val thisIndex = broadcastIndex(i, resultShape, this.shape)
            val otherIndex = broadcastIndex(i, resultShape, other.shape)
            result[i] = clampToInt(this.data[thisIndex].toLong() * other.data[otherIndex].toLong())
        }
        return CpuTensorInt32(resultShape, result)
    }

    override fun Tensor<Int32, Int>.div(other: Tensor<Int32, Int>): Tensor<Int32, Int> {
        require(this is CpuTensorInt32 && other is CpuTensorInt32) { "Both tensors must be CpuTensorInt32" }
        require(canBroadcast(this.shape, other.shape)) { "Tensors are not broadcast-compatible for element-wise division: ${this.shape} vs ${other.shape}" }

        val resultShape = getBroadcastShape(this.shape, other.shape)
        val result = IntArray(resultShape.volume)
        
        for (i in result.indices) {
            val thisIndex = broadcastIndex(i, resultShape, this.shape)
            val otherIndex = broadcastIndex(i, resultShape, other.shape)
            result[i] = if (other.data[otherIndex] != 0) this.data[thisIndex] / other.data[otherIndex] else 0
        }
        return CpuTensorInt32(resultShape, result)
    }

    // Tensor-Scalar operations
    override fun Tensor<Int32, Int>.plus(scalar: Int): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val result = this.data.map { clampToInt(it.toLong() + scalar.toLong()) }.toIntArray()
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.minus(scalar: Int): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val result = this.data.map { clampToInt(it.toLong() - scalar.toLong()) }.toIntArray()
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.times(scalar: Int): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val result = this.data.map { clampToInt(it.toLong() * scalar.toLong()) }.toIntArray()
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.div(scalar: Int): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val result = this.data.map { if (scalar != 0) it / scalar else 0 }.toIntArray()
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.plus(scalar: Float): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val result = this.data.map { clampToInt((it + scalar).toLong()) }.toIntArray()
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.minus(scalar: Float): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val result = this.data.map { clampToInt((it - scalar).toLong()) }.toIntArray()
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.times(scalar: Float): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val result = this.data.map { clampToInt((it * scalar).toLong()) }.toIntArray()
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.div(scalar: Float): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val result = this.data.map { if (scalar != 0f) clampToInt((it / scalar).toLong()) else 0 }.toIntArray()
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.plus(scalar: Double): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val result = this.data.map { clampToInt((it + scalar).toLong()) }.toIntArray()
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.minus(scalar: Double): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val result = this.data.map { clampToInt((it - scalar).toLong()) }.toIntArray()
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.times(scalar: Double): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val result = this.data.map { clampToInt((it * scalar).toLong()) }.toIntArray()
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.div(scalar: Double): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val result = this.data.map { if (scalar != 0.0) clampToInt((it / scalar).toLong()) else 0 }.toIntArray()
        return CpuTensorInt32(this.shape, result)
    }

    // Scalar-Tensor operations
    override fun Double.plus(t: Tensor<Int32, Int>): Tensor<Int32, Int> {
        require(t is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val result = t.data.map { clampToInt((this + it).toLong()) }.toIntArray()
        return CpuTensorInt32(t.shape, result)
    }

    override fun Double.minus(t: Tensor<Int32, Int>): Tensor<Int32, Int> {
        require(t is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val result = t.data.map { clampToInt((this - it).toLong()) }.toIntArray()
        return CpuTensorInt32(t.shape, result)
    }

    override fun Double.times(t: Tensor<Int32, Int>): Tensor<Int32, Int> {
        require(t is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val result = t.data.map { clampToInt((this * it).toLong()) }.toIntArray()
        return CpuTensorInt32(t.shape, result)
    }

    override fun Double.div(t: Tensor<Int32, Int>): Tensor<Int32, Int> {
        require(t is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val result = t.data.map { if (it != 0) clampToInt((this / it).toLong()) else 0 }.toIntArray()
        return CpuTensorInt32(t.shape, result)
    }

    // Advanced tensor operations
    override fun Tensor<Int32, Int>.t(): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        require(this.shape.rank == 2) { "Transpose only supported for 2D tensors (matrices)" }
        
        val rows = this.shape[0]
        val cols = this.shape[1]
        val result = IntArray(rows * cols)
        
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result[j * rows + i] = this.data[i * cols + j]
            }
        }
        
        return CpuTensorInt32(Shape(cols, rows), result)
    }

    override fun Tensor<Int32, Int>.relu(): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val result = this.data.map { maxOf(0, it) }.toIntArray()
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.sigmoid(): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        // For integer tensors, sigmoid is approximated and scaled to reasonable range
        val result = this.data.map { 
            val sigmoid = 1.0 / (1.0 + exp(-it.toDouble() / 2147483647.0))  // Scale input
            clampToInt((sigmoid * 2147483647.0).toLong())  // Scale output to positive int range
        }.toIntArray()
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.tanh(): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        // For integer tensors, tanh is approximated and scaled to int range
        val result = this.data.map { 
            val tanhValue = tanh(it.toDouble() / 2147483647.0)  // Scale input
            clampToInt((tanhValue * 2147483647.0).toLong())  // Scale output to int range
        }.toIntArray()
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.softmax(dimension: Int): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        require(dimension in 0 until this.shape.rank) { "Dimension $dimension is out of bounds for tensor with rank ${this.shape.rank}" }
        
        when (this.shape.rank) {
            1 -> {
                // Convert to double for softmax computation, then scale back
                val doubleData = this.data.map { it.toDouble() / 2147483647.0 }  // Scale to [-1, 1]
                val maxVal = doubleData.maxOrNull() ?: 0.0
                val expValues = doubleData.map { exp(it - maxVal) }
                val sum = expValues.sum()
                val result = expValues.map { clampToInt(((it / sum) * 2147483647.0).toLong()) }.toIntArray()
                return CpuTensorInt32(this.shape, result)
            }
            2 -> {
                val rows = this.shape[0]
                val cols = this.shape[1]
                val result = IntArray(this.data.size)
                
                if (dimension == 0) {
                    // Apply softmax along rows (for each column)
                    for (j in 0 until cols) {
                        val columnValues = DoubleArray(rows) { i -> this.data[i * cols + j].toDouble() / 2147483647.0 }
                        val maxVal = columnValues.maxOrNull() ?: 0.0
                        val expValues = columnValues.map { exp(it - maxVal) }
                        val sum = expValues.sum()
                        for (i in 0 until rows) {
                            result[i * cols + j] = clampToInt(((expValues[i] / sum) * 2147483647.0).toLong())
                        }
                    }
                } else {
                    // Apply softmax along columns (for each row)
                    for (i in 0 until rows) {
                        val rowStart = i * cols
                        val rowValues = IntArray(cols) { j -> this.data[rowStart + j] }.map { it.toDouble() / 2147483647.0 }
                        val maxVal = rowValues.maxOrNull() ?: 0.0
                        val expValues = rowValues.map { exp(it - maxVal) }
                        val sum = expValues.sum()
                        for (j in 0 until cols) {
                            result[rowStart + j] = clampToInt(((expValues[j] / sum) * 2147483647.0).toLong())
                        }
                    }
                }
                return CpuTensorInt32(this.shape, result)
            }
            else -> {
                throw UnsupportedOperationException("Softmax not implemented for tensors with rank > 2")
            }
        }
    }

    override fun Tensor<Int32, Int>.flatten(startDim: Int, endDim: Int): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        
        val actualEndDim = if (endDim == -1) this.shape.rank - 1 else endDim
        require(startDim >= 0 && startDim < this.shape.rank) { "startDim $startDim is out of bounds" }
        require(actualEndDim >= startDim && actualEndDim < this.shape.rank) { "endDim $actualEndDim is out of bounds or less than startDim" }
        
        if (startDim == actualEndDim) {
            return CpuTensorInt32(this.shape, this.data.copyOf())
        }
        
        val newDimensions = mutableListOf<Int>()
        
        for (i in 0 until startDim) {
            newDimensions.add(this.shape[i])
        }
        
        var flattenedSize = 1
        for (i in startDim..actualEndDim) {
            flattenedSize *= this.shape[i]
        }
        newDimensions.add(flattenedSize)
        
        for (i in (actualEndDim + 1) until this.shape.rank) {
            newDimensions.add(this.shape[i])
        }
        
        val newShape = Shape(*newDimensions.toIntArray())
        return CpuTensorInt32(newShape, this.data.copyOf())
    }

    public companion object {
        /**
         * Creates a tensor from an array with the given shape.
         */
        public fun fromArray(shape: Shape, data: IntArray): CpuTensorInt32 {
            return CpuTensorInt32(shape, data)
        }

        /**
         * Creates a tensor filled with zeros.
         */
        public fun zeros(shape: Shape): CpuTensorInt32 {
            return CpuTensorInt32(shape, IntArray(shape.volume))
        }

        /**
         * Creates a tensor filled with ones.
         */
        public fun ones(shape: Shape): CpuTensorInt32 {
            return CpuTensorInt32(shape, IntArray(shape.volume) { 1 })
        }

        /**
         * Creates a tensor filled with a specific value.
         */
        public fun full(shape: Shape, value: Int): CpuTensorInt32 {
            return CpuTensorInt32(shape, IntArray(shape.volume) { value })
        }
    }
}

// Broadcasting utility functions
private fun canBroadcast(shape1: Shape, shape2: Shape): Boolean {
    val rank1 = shape1.rank
    val rank2 = shape2.rank
    val maxRank = maxOf(rank1, rank2)
    
    for (i in 0 until maxRank) {
        val dim1 = if (i < rank1) shape1[rank1 - 1 - i] else 1
        val dim2 = if (i < rank2) shape2[rank2 - 1 - i] else 1
        
        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            return false
        }
    }
    return true
}

private fun getBroadcastShape(shape1: Shape, shape2: Shape): Shape {
    val rank1 = shape1.rank
    val rank2 = shape2.rank
    val maxRank = maxOf(rank1, rank2)
    val resultDims = IntArray(maxRank)
    
    for (i in 0 until maxRank) {
        val dim1 = if (i < rank1) shape1[rank1 - 1 - i] else 1
        val dim2 = if (i < rank2) shape2[rank2 - 1 - i] else 1
        resultDims[maxRank - 1 - i] = maxOf(dim1, dim2)
    }
    
    return Shape(*resultDims)
}

private fun broadcastIndex(flatIndex: Int, resultShape: Shape, originalShape: Shape): Int {
    val resultRank = resultShape.rank
    val originalRank = originalShape.rank
    
    // Convert flat index to multi-dimensional indices for result shape
    val resultIndices = IntArray(resultRank)
    var temp = flatIndex
    for (i in resultRank - 1 downTo 0) {
        resultIndices[i] = temp % resultShape[i]
        temp /= resultShape[i]
    }
    
    // Map to original shape indices (handle broadcasting)
    val originalIndices = IntArray(originalRank)
    for (i in 0 until originalRank) {
        val resultI = resultRank - originalRank + i
        if (resultI >= 0) {
            val originalDim = originalShape[i]
            val resultDim = resultShape[resultI]
            originalIndices[i] = if (originalDim == 1 && resultDim > 1) 0 else resultIndices[resultI]
        }
    }
    
    // Convert back to flat index for original shape
    var originalIndex = 0
    for (i in 0 until originalRank) {
        originalIndex = originalIndex * originalShape[i] + originalIndices[i]
    }
    
    return originalIndex
}

/**
 * A CPU-based implementation of the ComputeBackend interface for FP32/Float tensors.
 */
public class CpuBackend : ComputeBackend<FP32, Float> {
    override val name: String = "CPU"
    
    init {
        TensorFactoryInitializer.ensureInitialized()
    }

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
        require(canBroadcast(this.shape, other.shape)) { "Tensors are not broadcast-compatible for addition: ${this.shape} vs ${other.shape}" }

        val resultShape = getBroadcastShape(this.shape, other.shape)
        val result = FloatArray(resultShape.volume)
        
        for (i in result.indices) {
            val thisIndex = broadcastIndex(i, resultShape, this.shape)
            val otherIndex = broadcastIndex(i, resultShape, other.shape)
            result[i] = this.data[thisIndex] + other.data[otherIndex]
        }
        return CpuTensorFP32(resultShape, result)
    }

    override fun Tensor<FP32, Float>.minus(other: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(this is CpuTensorFP32 && other is CpuTensorFP32) { "Both tensors must be CpuTensorFP32" }
        require(canBroadcast(this.shape, other.shape)) { "Tensors are not broadcast-compatible for subtraction: ${this.shape} vs ${other.shape}" }

        val resultShape = getBroadcastShape(this.shape, other.shape)
        val result = FloatArray(resultShape.volume)
        
        for (i in result.indices) {
            val thisIndex = broadcastIndex(i, resultShape, this.shape)
            val otherIndex = broadcastIndex(i, resultShape, other.shape)
            result[i] = this.data[thisIndex] - other.data[otherIndex]
        }
        return CpuTensorFP32(resultShape, result)
    }

    override fun Tensor<FP32, Float>.times(other: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(this is CpuTensorFP32 && other is CpuTensorFP32) { "Both tensors must be CpuTensorFP32" }
        require(canBroadcast(this.shape, other.shape)) { "Tensors are not broadcast-compatible for element-wise multiplication: ${this.shape} vs ${other.shape}" }

        val resultShape = getBroadcastShape(this.shape, other.shape)
        val result = FloatArray(resultShape.volume)
        
        for (i in result.indices) {
            val thisIndex = broadcastIndex(i, resultShape, this.shape)
            val otherIndex = broadcastIndex(i, resultShape, other.shape)
            result[i] = this.data[thisIndex] * other.data[otherIndex]
        }
        return CpuTensorFP32(resultShape, result)
    }

    override fun Tensor<FP32, Float>.div(other: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(this is CpuTensorFP32 && other is CpuTensorFP32) { "Both tensors must be CpuTensorFP32" }
        require(canBroadcast(this.shape, other.shape)) { "Tensors are not broadcast-compatible for element-wise division: ${this.shape} vs ${other.shape}" }

        val resultShape = getBroadcastShape(this.shape, other.shape)
        val result = FloatArray(resultShape.volume)
        
        for (i in result.indices) {
            val thisIndex = broadcastIndex(i, resultShape, this.shape)
            val otherIndex = broadcastIndex(i, resultShape, other.shape)
            result[i] = this.data[thisIndex] / other.data[otherIndex]
        }
        return CpuTensorFP32(resultShape, result)
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

    // TensorFactory interface implementation
    override fun zeros(shape: Shape): Tensor<FP32, Float> {
        return CpuTensorFP32.zeros(shape)
    }

    override fun ones(shape: Shape): Tensor<FP32, Float> {
        return CpuTensorFP32.ones(shape)
    }
}

/**
 * A CPU-based implementation of the ComputeBackend interface for Int8/Byte tensors.
 */
public class CpuBackendInt8 : ComputeBackend<Int8, Byte> {
    override val name: String = "CPU-Int8"
    
    init {
        TensorFactoryInitializer.ensureInitialized()
    }

    override fun matmul(a: Tensor<Int8, Byte>, b: Tensor<Int8, Byte>): Tensor<Int8, Byte> =
        (a as CpuTensorInt8).matmul(a, b)

    override fun scale(a: Tensor<Int8, Byte>, scalar: Double): Tensor<Int8, Byte> =
        (a as CpuTensorInt8).scale(a, scalar)

    override fun dot(a: Tensor<Int8, Byte>, b: Tensor<Int8, Byte>): Double =
        (a as CpuTensorInt8).dot(a, b)

    override fun Tensor<Int8, Byte>.plus(other: Tensor<Int8, Byte>): Tensor<Int8, Byte> =
        (this as CpuTensorInt8).plus(other)

    override fun Tensor<Int8, Byte>.minus(other: Tensor<Int8, Byte>): Tensor<Int8, Byte> =
        (this as CpuTensorInt8).minus(other)

    override fun Tensor<Int8, Byte>.times(other: Tensor<Int8, Byte>): Tensor<Int8, Byte> =
        (this as CpuTensorInt8).times(other)

    override fun Tensor<Int8, Byte>.div(other: Tensor<Int8, Byte>): Tensor<Int8, Byte> =
        (this as CpuTensorInt8).div(other)

    override fun Tensor<Int8, Byte>.plus(scalar: Int): Tensor<Int8, Byte> =
        (this as CpuTensorInt8).plus(scalar)

    override fun Tensor<Int8, Byte>.minus(scalar: Int): Tensor<Int8, Byte> =
        (this as CpuTensorInt8).minus(scalar)

    override fun Tensor<Int8, Byte>.times(scalar: Int): Tensor<Int8, Byte> =
        (this as CpuTensorInt8).times(scalar)

    override fun Tensor<Int8, Byte>.div(scalar: Int): Tensor<Int8, Byte> =
        (this as CpuTensorInt8).div(scalar)

    override fun Tensor<Int8, Byte>.plus(scalar: Float): Tensor<Int8, Byte> =
        (this as CpuTensorInt8).plus(scalar)

    override fun Tensor<Int8, Byte>.minus(scalar: Float): Tensor<Int8, Byte> =
        (this as CpuTensorInt8).minus(scalar)

    override fun Tensor<Int8, Byte>.times(scalar: Float): Tensor<Int8, Byte> =
        (this as CpuTensorInt8).times(scalar)

    override fun Tensor<Int8, Byte>.div(scalar: Float): Tensor<Int8, Byte> =
        (this as CpuTensorInt8).div(scalar)

    override fun Tensor<Int8, Byte>.plus(scalar: Double): Tensor<Int8, Byte> =
        (this as CpuTensorInt8).plus(scalar)

    override fun Tensor<Int8, Byte>.minus(scalar: Double): Tensor<Int8, Byte> =
        (this as CpuTensorInt8).minus(scalar)

    override fun Tensor<Int8, Byte>.times(scalar: Double): Tensor<Int8, Byte> =
        (this as CpuTensorInt8).times(scalar)

    override fun Tensor<Int8, Byte>.div(scalar: Double): Tensor<Int8, Byte> =
        (this as CpuTensorInt8).div(scalar)

    override fun Double.plus(t: Tensor<Int8, Byte>): Tensor<Int8, Byte> =
        (t as CpuTensorInt8).plus(this)

    override fun Double.minus(t: Tensor<Int8, Byte>): Tensor<Int8, Byte> =
        (t as CpuTensorInt8).minus(this).let {
            CpuTensorInt8.fromArray(t.shape, ByteArray(t.shape.volume) { i ->
                ((this - (t as CpuTensorInt8).data[i].toInt()).toInt().toByte())
            })
        }

    override fun Double.times(t: Tensor<Int8, Byte>): Tensor<Int8, Byte> =
        (t as CpuTensorInt8).times(this)

    override fun Double.div(t: Tensor<Int8, Byte>): Tensor<Int8, Byte> =
        CpuTensorInt8.fromArray(t.shape, ByteArray(t.shape.volume) { i ->
            ((this / (t as CpuTensorInt8).data[i].toInt()).toInt().toByte())
        })

    override fun Tensor<Int8, Byte>.t(): Tensor<Int8, Byte> =
        (this as CpuTensorInt8).t()

    override fun Tensor<Int8, Byte>.relu(): Tensor<Int8, Byte> =
        (this as CpuTensorInt8).relu()

    override fun Tensor<Int8, Byte>.sigmoid(): Tensor<Int8, Byte> =
        (this as CpuTensorInt8).sigmoid()

    override fun Tensor<Int8, Byte>.tanh(): Tensor<Int8, Byte> =
        (this as CpuTensorInt8).tanh()

    override fun Tensor<Int8, Byte>.softmax(dimension: Int): Tensor<Int8, Byte> =
        (this as CpuTensorInt8).softmax(dimension)

    override fun Tensor<Int8, Byte>.flatten(startDim: Int, endDim: Int): Tensor<Int8, Byte> =
        (this as CpuTensorInt8).flatten(startDim, endDim)

    override fun zeros(shape: Shape): Tensor<Int8, Byte> =
        CpuTensorInt8.zeros(shape)

    override fun ones(shape: Shape): Tensor<Int8, Byte> =
        CpuTensorInt8.ones(shape)
}

/**
 * A CPU-based implementation of the ComputeBackend interface for Int32/Int tensors.
 */
public class CpuBackendInt32 : ComputeBackend<Int32, Int> {
    override val name: String = "CPU-Int32"
    
    init {
        TensorFactoryInitializer.ensureInitialized()
    }

    override fun matmul(a: Tensor<Int32, Int>, b: Tensor<Int32, Int>): Tensor<Int32, Int> =
        (a as CpuTensorInt32).matmul(a, b)

    override fun scale(a: Tensor<Int32, Int>, scalar: Double): Tensor<Int32, Int> =
        (a as CpuTensorInt32).scale(a, scalar)

    override fun dot(a: Tensor<Int32, Int>, b: Tensor<Int32, Int>): Double =
        (a as CpuTensorInt32).dot(a, b)

    override fun Tensor<Int32, Int>.plus(other: Tensor<Int32, Int>): Tensor<Int32, Int> =
        (this as CpuTensorInt32).plus(other)

    override fun Tensor<Int32, Int>.minus(other: Tensor<Int32, Int>): Tensor<Int32, Int> =
        (this as CpuTensorInt32).minus(other)

    override fun Tensor<Int32, Int>.times(other: Tensor<Int32, Int>): Tensor<Int32, Int> =
        (this as CpuTensorInt32).times(other)

    override fun Tensor<Int32, Int>.div(other: Tensor<Int32, Int>): Tensor<Int32, Int> =
        (this as CpuTensorInt32).div(other)

    override fun Tensor<Int32, Int>.plus(scalar: Int): Tensor<Int32, Int> =
        (this as CpuTensorInt32).plus(scalar)

    override fun Tensor<Int32, Int>.minus(scalar: Int): Tensor<Int32, Int> =
        (this as CpuTensorInt32).minus(scalar)

    override fun Tensor<Int32, Int>.times(scalar: Int): Tensor<Int32, Int> =
        (this as CpuTensorInt32).times(scalar)

    override fun Tensor<Int32, Int>.div(scalar: Int): Tensor<Int32, Int> =
        (this as CpuTensorInt32).div(scalar)

    override fun Tensor<Int32, Int>.plus(scalar: Float): Tensor<Int32, Int> =
        (this as CpuTensorInt32).plus(scalar)

    override fun Tensor<Int32, Int>.minus(scalar: Float): Tensor<Int32, Int> =
        (this as CpuTensorInt32).minus(scalar)

    override fun Tensor<Int32, Int>.times(scalar: Float): Tensor<Int32, Int> =
        (this as CpuTensorInt32).times(scalar)

    override fun Tensor<Int32, Int>.div(scalar: Float): Tensor<Int32, Int> =
        (this as CpuTensorInt32).div(scalar)

    override fun Tensor<Int32, Int>.plus(scalar: Double): Tensor<Int32, Int> =
        (this as CpuTensorInt32).plus(scalar)

    override fun Tensor<Int32, Int>.minus(scalar: Double): Tensor<Int32, Int> =
        (this as CpuTensorInt32).minus(scalar)

    override fun Tensor<Int32, Int>.times(scalar: Double): Tensor<Int32, Int> =
        (this as CpuTensorInt32).times(scalar)

    override fun Tensor<Int32, Int>.div(scalar: Double): Tensor<Int32, Int> =
        (this as CpuTensorInt32).div(scalar)

    override fun Double.plus(t: Tensor<Int32, Int>): Tensor<Int32, Int> =
        (t as CpuTensorInt32).plus(this)

    override fun Double.minus(t: Tensor<Int32, Int>): Tensor<Int32, Int> =
        CpuTensorInt32.fromArray(t.shape, IntArray(t.shape.volume) { i ->
            (this - (t as CpuTensorInt32).data[i]).toInt()
        })

    override fun Double.times(t: Tensor<Int32, Int>): Tensor<Int32, Int> =
        (t as CpuTensorInt32).times(this)

    override fun Double.div(t: Tensor<Int32, Int>): Tensor<Int32, Int> =
        CpuTensorInt32.fromArray(t.shape, IntArray(t.shape.volume) { i ->
            (this / (t as CpuTensorInt32).data[i]).toInt()
        })

    override fun Tensor<Int32, Int>.t(): Tensor<Int32, Int> =
        (this as CpuTensorInt32).t()

    override fun Tensor<Int32, Int>.relu(): Tensor<Int32, Int> =
        (this as CpuTensorInt32).relu()

    override fun Tensor<Int32, Int>.sigmoid(): Tensor<Int32, Int> =
        (this as CpuTensorInt32).sigmoid()

    override fun Tensor<Int32, Int>.tanh(): Tensor<Int32, Int> =
        (this as CpuTensorInt32).tanh()

    override fun Tensor<Int32, Int>.softmax(dimension: Int): Tensor<Int32, Int> =
        (this as CpuTensorInt32).softmax(dimension)

    override fun Tensor<Int32, Int>.flatten(startDim: Int, endDim: Int): Tensor<Int32, Int> =
        (this as CpuTensorInt32).flatten(startDim, endDim)

    override fun zeros(shape: Shape): Tensor<Int32, Int> =
        CpuTensorInt32.zeros(shape)

    override fun ones(shape: Shape): Tensor<Int32, Int> =
        CpuTensorInt32.ones(shape)
}