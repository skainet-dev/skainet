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
 * Convenient type alias for Int4 tensors with Byte values.
 * Use this instead of concrete implementations where possible for better abstraction.
 */
public typealias TensorInt4 = Tensor<Int4, Byte>

/**
 * Convenient type alias for Ternary tensors with Byte values.
 * Use this instead of concrete implementations where possible for better abstraction.
 */
public typealias TensorTernary = Tensor<Ternary, Byte>

/**
 * Convenient type alias for FP16 tensors with Float values.
 * Use this instead of concrete implementations where possible for better abstraction.
 */
public typealias TensorFP16 = Tensor<FP16, Float>

/**
 * A CPU-based tensor for FP32/Float values.
 *
 * This tensor stores data on the CPU using TensorData abstraction with NCHW row-major layout.
 * It supports 1-4 dimensional tensors and delegates all operations to CpuBackend.
 */
public class CpuTensorFP32(
    private val tensorData: TensorData<FP32, Float>
) : TensorFP32 {

    // Legacy constructor for backward compatibility
    public constructor(shape: Shape, data: FloatArray) : this(
        DenseTensorData<FP32, Float>(shape, data.toTypedArray())
    )

    override val shape: Shape get() = tensorData.shape
    override val strides: IntArray get() = tensorData.strides
    override val offset: Int get() = tensorData.offset
    override val isContiguous: Boolean get() = tensorData.isContiguous

    init {
        require(shape.rank in 1..4) {
            "Only 1-4 dimensional tensors are supported, got ${shape.rank}"
        }
    }

    override fun get(vararg indices: Int): Float = tensorData.get(*indices)

    override fun copyTo(dest: Array<Float>, destOffset: Int): Unit = tensorData.copyTo(dest, destOffset)
    
    override fun slice(ranges: IntArray): TensorData<FP32, Float> = tensorData.slice(ranges)
    
    override fun materialize(): TensorData<FP32, Float> = tensorData.materialize()

    // Internal data access for CpuBackend operations
    internal val data: FloatArray get() {
        return when (tensorData) {
            is DenseTensorData -> {
                // Extract FloatArray from DenseTensorData
                val floatArray = FloatArray(shape.volume)
                val arrayFloat = Array<Float>(shape.volume) { 0f }
                tensorData.copyTo(arrayFloat, 0)
                for (i in arrayFloat.indices) {
                    floatArray[i] = arrayFloat[i]
                }
                floatArray
            }
            else -> {
                // Materialize other tensor data types to FloatArray
                val materialized = tensorData.materialize()
                val floatArray = FloatArray(shape.volume)
                val arrayFloat = Array<Float>(shape.volume) { 0f }
                materialized.copyTo(arrayFloat, 0)
                for (i in arrayFloat.indices) {
                    floatArray[i] = arrayFloat[i]
                }
                floatArray
            }
        }
    }

    // Delegate all operations to the CpuBackend
    private val backend = CpuBackend()

    override fun matmul(a: Tensor<FP32, Float>, b: Tensor<FP32, Float>): Tensor<FP32, Float> = backend.matmul(a, b)
    override fun matmul4d(a: Tensor<FP32, Float>, b: Tensor<FP32, Float>): Tensor<FP32, Float> = backend.matmul4d(a, b)
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
    override fun Tensor<FP32, Float>.reshape(newShape: Shape): Tensor<FP32, Float> = with(backend) { this@reshape.reshape(newShape) }
    override fun Tensor<FP32, Float>.reshape(vararg dimensions: Int): Tensor<FP32, Float> = with(backend) { this@reshape.reshape(*dimensions) }

    public companion object {
        /**
         * Creates a tensor from an array with the given shape.
         */
        public fun fromArray(shape: Shape, data: FloatArray): CpuTensorFP32 {
            require(data.size == shape.volume) {
                "Data size ${data.size} doesn't match shape volume ${shape.volume}"
            }
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

    // TensorData implementation
    override val strides: IntArray = shape.computeStrides()
    override val offset: Int = 0
    override val isContiguous: Boolean = true
    
    override fun copyTo(dest: Array<Byte>, destOffset: Int) {
        for (i in data.indices) {
            dest[destOffset + i] = data[i]
        }
    }
    
    override fun slice(ranges: IntArray): TensorData<Int8, Byte> {
        require(ranges.size == shape.rank * 2) {
            "Ranges array must have size ${shape.rank * 2} (start,end pairs), got ${ranges.size}"
        }
        
        // Parse ranges into start/end pairs and validate
        val sliceRanges = mutableListOf<Pair<Int, Int>>()
        for (i in 0 until shape.rank) {
            val start = ranges[i * 2]
            val end = ranges[i * 2 + 1]
            val dimSize = shape.dimensions[i]
            
            require(start >= 0 && start < dimSize) {
                "Start index $start out of bounds for dimension $i (size $dimSize)"
            }
            require(end > start && end <= dimSize) {
                "End index $end must be > start ($start) and <= dimension size ($dimSize)"
            }
            
            sliceRanges.add(start to end)
        }
        
        // Calculate new shape
        val newDimensions = sliceRanges.map { (start, end) -> end - start }.toIntArray()
        val newShape = Shape(newDimensions)
        
        // Create new data array
        val newData = ByteArray(newShape.volume)
        var destIndex = 0
        
        // Copy sliced data
        when (shape.rank) {
            1 -> {
                val (start0, end0) = sliceRanges[0]
                for (i in start0 until end0) {
                    newData[destIndex++] = data[i]
                }
            }
            2 -> {
                val (start0, end0) = sliceRanges[0]
                val (start1, end1) = sliceRanges[1]
                for (i in start0 until end0) {
                    for (j in start1 until end1) {
                        val srcIndex = i * shape.dimensions[1] + j
                        newData[destIndex++] = data[srcIndex]
                    }
                }
            }
            3 -> {
                val (start0, end0) = sliceRanges[0]
                val (start1, end1) = sliceRanges[1]
                val (start2, end2) = sliceRanges[2]
                for (i in start0 until end0) {
                    for (j in start1 until end1) {
                        for (k in start2 until end2) {
                            val srcIndex = i * shape.dimensions[1] * shape.dimensions[2] +
                                         j * shape.dimensions[2] + k
                            newData[destIndex++] = data[srcIndex]
                        }
                    }
                }
            }
            4 -> {
                val (start0, end0) = sliceRanges[0]
                val (start1, end1) = sliceRanges[1]
                val (start2, end2) = sliceRanges[2]
                val (start3, end3) = sliceRanges[3]
                for (i in start0 until end0) {
                    for (j in start1 until end1) {
                        for (k in start2 until end2) {
                            for (l in start3 until end3) {
                                val srcIndex = i * shape.dimensions[1] * shape.dimensions[2] * shape.dimensions[3] +
                                             j * shape.dimensions[2] * shape.dimensions[3] +
                                             k * shape.dimensions[3] + l
                                newData[destIndex++] = data[srcIndex]
                            }
                        }
                    }
                }
            }
            else -> throw UnsupportedOperationException("Slicing not supported for ${shape.rank}D tensors")
        }
        
        return CpuTensorInt8(newShape, newData)
    }
    
    override fun materialize(): TensorData<Int8, Byte> = this

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
    
    private fun Shape.computeStrides(): IntArray {
        if (dimensions.isEmpty()) return intArrayOf()
        val strides = IntArray(dimensions.size)
        strides[dimensions.size - 1] = 1
        for (i in dimensions.size - 2 downTo 0) {
            strides[i] = strides[i + 1] * dimensions[i + 1]
        }
        return strides
    }

    // Helper function to clamp values to Byte range
    private fun clampToByte(value: Double): Byte = value.toInt().coerceIn(Byte.MIN_VALUE.toInt(), Byte.MAX_VALUE.toInt()).toByte()

    // Basic operations - implement directly
    override fun matmul4d(a: Tensor<Int8, Byte>, b: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        throw UnsupportedOperationException("4D matrix multiplication not yet implemented for Int8 tensors")
    }

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

    override fun Tensor<Int8, Byte>.reshape(newShape: Shape): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        require(this.shape.volume == newShape.volume) {
            "Cannot reshape tensor with ${this.shape.volume} elements to shape with ${newShape.volume} elements"
        }
        
        // Create new tensor with same data but new shape
        return CpuTensorInt8(newShape, this.data.copyOf())
    }
    
    override fun Tensor<Int8, Byte>.reshape(vararg dimensions: Int): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        
        // Count -1 dimensions and validate
        val minusOneCount = dimensions.count { it == -1 }
        require(minusOneCount <= 1) { "Only one dimension can be -1, found $minusOneCount" }
        
        if (minusOneCount == 0) {
            // No -1, use regular reshape
            return this.reshape(Shape(*dimensions))
        }
        
        // Calculate inferred dimension
        val minusOneIndex = dimensions.indexOfFirst { it == -1 }
        val knownProduct = dimensions.filter { it != -1 }.fold(1) { acc, dim ->
            require(dim > 0) { "All dimensions except -1 must be positive, got $dim" }
            acc * dim
        }
        
        require(this.shape.volume % knownProduct == 0) {
            "Cannot reshape tensor with ${this.shape.volume} elements: remaining elements $knownProduct do not divide evenly"
        }
        
        val inferredDim = this.shape.volume / knownProduct
        val newDimensions = dimensions.copyOf()
        newDimensions[minusOneIndex] = inferredDim
        
        return this.reshape(Shape(*newDimensions))
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

    // TensorData implementation
    override val strides: IntArray = shape.computeStrides()
    override val offset: Int = 0
    override val isContiguous: Boolean = true
    
    override fun copyTo(dest: Array<Int>, destOffset: Int) {
        for (i in data.indices) {
            dest[destOffset + i] = data[i]
        }
    }
    
    override fun slice(ranges: IntArray): TensorData<Int32, Int> {
        require(ranges.size == shape.rank * 2) {
            "Ranges array must have size ${shape.rank * 2} (start,end pairs), got ${ranges.size}"
        }
        
        // Parse ranges into start/end pairs and validate
        val sliceRanges = mutableListOf<Pair<Int, Int>>()
        for (i in 0 until shape.rank) {
            val start = ranges[i * 2]
            val end = ranges[i * 2 + 1]
            val dimSize = shape.dimensions[i]
            
            require(start >= 0 && start < dimSize) {
                "Start index $start out of bounds for dimension $i (size $dimSize)"
            }
            require(end > start && end <= dimSize) {
                "End index $end must be > start ($start) and <= dimension size ($dimSize)"
            }
            
            sliceRanges.add(start to end)
        }
        
        // Calculate new shape
        val newDimensions = sliceRanges.map { (start, end) -> end - start }.toIntArray()
        val newShape = Shape(newDimensions)
        
        // Create new data array
        val newData = IntArray(newShape.volume)
        var destIndex = 0
        
        // Copy sliced data
        when (shape.rank) {
            1 -> {
                val (start0, end0) = sliceRanges[0]
                for (i in start0 until end0) {
                    newData[destIndex++] = data[i]
                }
            }
            2 -> {
                val (start0, end0) = sliceRanges[0]
                val (start1, end1) = sliceRanges[1]
                for (i in start0 until end0) {
                    for (j in start1 until end1) {
                        val srcIndex = i * shape.dimensions[1] + j
                        newData[destIndex++] = data[srcIndex]
                    }
                }
            }
            3 -> {
                val (start0, end0) = sliceRanges[0]
                val (start1, end1) = sliceRanges[1]
                val (start2, end2) = sliceRanges[2]
                for (i in start0 until end0) {
                    for (j in start1 until end1) {
                        for (k in start2 until end2) {
                            val srcIndex = i * shape.dimensions[1] * shape.dimensions[2] +
                                         j * shape.dimensions[2] + k
                            newData[destIndex++] = data[srcIndex]
                        }
                    }
                }
            }
            4 -> {
                val (start0, end0) = sliceRanges[0]
                val (start1, end1) = sliceRanges[1]
                val (start2, end2) = sliceRanges[2]
                val (start3, end3) = sliceRanges[3]
                for (i in start0 until end0) {
                    for (j in start1 until end1) {
                        for (k in start2 until end2) {
                            for (l in start3 until end3) {
                                val srcIndex = i * shape.dimensions[1] * shape.dimensions[2] * shape.dimensions[3] +
                                             j * shape.dimensions[2] * shape.dimensions[3] +
                                             k * shape.dimensions[3] + l
                                newData[destIndex++] = data[srcIndex]
                            }
                        }
                    }
                }
            }
            else -> throw UnsupportedOperationException("Slicing not supported for ${shape.rank}D tensors")
        }
        
        return CpuTensorInt32(newShape, newData)
    }
    
    override fun materialize(): TensorData<Int32, Int> = this

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
    
    private fun Shape.computeStrides(): IntArray {
        if (dimensions.isEmpty()) return intArrayOf()
        val strides = IntArray(dimensions.size)
        strides[dimensions.size - 1] = 1
        for (i in dimensions.size - 2 downTo 0) {
            strides[i] = strides[i + 1] * dimensions[i + 1]
        }
        return strides
    }

    // Helper function to clamp values to Int range (overflow protection)
    private fun clampToInt(value: Long): Int = value.coerceIn(Int.MIN_VALUE.toLong(), Int.MAX_VALUE.toLong()).toInt()

    // Basic operations - implement directly
    override fun matmul4d(a: Tensor<Int32, Int>, b: Tensor<Int32, Int>): Tensor<Int32, Int> {
        throw UnsupportedOperationException("4D matrix multiplication not yet implemented for Int32 tensors")
    }

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

    override fun Tensor<Int32, Int>.reshape(newShape: Shape): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        require(this.shape.volume == newShape.volume) {
            "Cannot reshape tensor with ${this.shape.volume} elements to shape with ${newShape.volume} elements"
        }
        
        // Create new tensor with same data but new shape
        return CpuTensorInt32(newShape, this.data.copyOf())
    }
    
    override fun Tensor<Int32, Int>.reshape(vararg dimensions: Int): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        
        // Count -1 dimensions and validate
        val minusOneCount = dimensions.count { it == -1 }
        require(minusOneCount <= 1) { "Only one dimension can be -1, found $minusOneCount" }
        
        if (minusOneCount == 0) {
            // No -1, use regular reshape
            return this.reshape(Shape(*dimensions))
        }
        
        // Calculate inferred dimension
        val minusOneIndex = dimensions.indexOfFirst { it == -1 }
        val knownProduct = dimensions.filter { it != -1 }.fold(1) { acc, dim ->
            require(dim > 0) { "All dimensions except -1 must be positive, got $dim" }
            acc * dim
        }
        
        require(this.shape.volume % knownProduct == 0) {
            "Cannot reshape tensor with ${this.shape.volume} elements: remaining elements $knownProduct do not divide evenly"
        }
        
        val inferredDim = this.shape.volume / knownProduct
        val newDimensions = dimensions.copyOf()
        newDimensions[minusOneIndex] = inferredDim
        
        return this.reshape(Shape(*newDimensions))
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

private fun flatIndexToCoords(flatIndex: Int, shape: Shape): IntArray {
    val coords = IntArray(shape.rank)
    var temp = flatIndex
    for (i in shape.rank - 1 downTo 0) {
        coords[i] = temp % shape[i]
        temp /= shape[i]
    }
    return coords
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
                    sum += a[i, k] * b[k, j]
                }
                result[i * cols + j] = sum
            }
        }

        return CpuTensorFP32(Shape(rows, cols), result)
    }

    override fun matmul4d(a: Tensor<FP32, Float>, b: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(a.shape.rank == 4) { "First tensor must be 4D for matmul4d, got ${a.shape.rank}D" }
        require(b.shape.rank == 4 || b.shape.rank == 2) { "Second tensor must be 4D or 2D for matmul4d, got ${b.shape.rank}D" }

        return when (b.shape.rank) {
            4 -> matmul4dBatch(a, b)
            2 -> matmul4dMixed(a, b)
            else -> throw IllegalArgumentException("Unsupported tensor rank combination")
        }
    }

    private fun matmul4dBatch(a: Tensor<FP32, Float>, b: Tensor<FP32, Float>): Tensor<FP32, Float> {
        // Batch-wise matrix multiplication: (B,M,K) × (B,K,N) → (B,M,N)
        // Also supports spatial: (B,C,H,W) × (B,C,W,K) → (B,C,H,K)
        
        val batchSize = a.shape[0]
        val channels = a.shape[1]
        val height = a.shape[2] 
        val width = a.shape[3]
        
        require(a.shape[0] == b.shape[0]) { "Batch dimensions must match: ${a.shape[0]} vs ${b.shape[0]}" }
        require(a.shape[1] == b.shape[1]) { "Channel dimensions must match: ${a.shape[1]} vs ${b.shape[1]}" }
        require(a.shape[3] == b.shape[2]) { "Inner dimensions must match for multiplication: ${a.shape[3]} vs ${b.shape[2]}" }
        
        val outputDim = b.shape[3]
        val resultShape = Shape(batchSize, channels, height, outputDim)
        val result = FloatArray(resultShape.volume)
        
        // Perform batch-wise matrix multiplication
        for (batch in 0 until batchSize) {
            for (channel in 0 until channels) {
                for (i in 0 until height) {
                    for (j in 0 until outputDim) {
                        var sum = 0f
                        for (k in 0 until width) {
                            sum += a[batch, channel, i, k] * b[batch, channel, k, j]
                        }
                        val resultIndex = batch * (channels * height * outputDim) + 
                                        channel * (height * outputDim) + 
                                        i * outputDim + j
                        result[resultIndex] = sum
                    }
                }
            }
        }
        
        return CpuTensorFP32(resultShape, result)
    }

    private fun matmul4dMixed(a: Tensor<FP32, Float>, b: Tensor<FP32, Float>): Tensor<FP32, Float> {
        // Mixed rank: (B,C,H,W) × (W,K) → (B,C,H,K) 
        // Useful for channel mixing: (B,C1,H,W) × (C1*H*W,C2) → (B,C2,H,W) when reshaped appropriately
        
        val batchSize = a.shape[0]
        val channels = a.shape[1]
        val height = a.shape[2]
        val width = a.shape[3]
        
        require(a.shape[3] == b.shape[0]) { "Inner dimensions must match: tensor A width ${a.shape[3]} vs matrix B rows ${b.shape[0]}" }
        
        val outputChannels = b.shape[1]
        val resultShape = Shape(batchSize, channels, height, outputChannels)
        val result = FloatArray(resultShape.volume)
        
        // Perform mixed-rank multiplication
        for (batch in 0 until batchSize) {
            for (channel in 0 until channels) {
                for (i in 0 until height) {
                    for (j in 0 until outputChannels) {
                        var sum = 0f
                        for (k in 0 until width) {
                            sum += a[batch, channel, i, k] * b[k, j]
                        }
                        val resultIndex = batch * (channels * height * outputChannels) + 
                                        channel * (height * outputChannels) + 
                                        i * outputChannels + j
                        result[resultIndex] = sum
                    }
                }
            }
        }
        
        return CpuTensorFP32(resultShape, result)
    }

    override fun scale(a: Tensor<FP32, Float>, scalar: Double): Tensor<FP32, Float> {
        val result = FloatArray(a.shape.volume)
        val coords = IntArray(a.shape.rank)
        var index = 0
        
        fun iterate(dim: Int) {
            if (dim == a.shape.rank) {
                result[index++] = a.get(*coords) * scalar.toFloat()
                return
            }
            for (i in 0 until a.shape[dim]) {
                coords[dim] = i
                iterate(dim + 1)
            }
        }
        
        iterate(0)
        return CpuTensorFP32(a.shape, result)
    }

    override fun dot(a: Tensor<FP32, Float>, b: Tensor<FP32, Float>): Double {
        require(a.shape == b.shape) { "Tensors must have same shape for dot product" }

        var sum = 0.0
        val coords = IntArray(a.shape.rank)
        
        fun iterate(dim: Int) {
            if (dim == a.shape.rank) {
                sum += a.get(*coords) * b.get(*coords)
                return
            }
            for (i in 0 until a.shape[dim]) {
                coords[dim] = i
                iterate(dim + 1)
            }
        }
        
        iterate(0)
        return sum
    }

    // Tensor-Tensor operations - stride-aware implementation
    override fun Tensor<FP32, Float>.plus(other: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(canBroadcast(this.shape, other.shape)) { "Tensors are not broadcast-compatible for addition: ${this.shape} vs ${other.shape}" }

        val resultShape = getBroadcastShape(this.shape, other.shape)
        val result = Array<Float>(resultShape.volume) { 0.0f }
        
        // Optimize for contiguous tensors of same shape
        if (this.isContiguous && other.isContiguous && this.shape == other.shape) {
            for (i in result.indices) {
                val coords = flatIndexToCoords(i, this.shape)
                result[i] = this.get(*coords) + other.get(*coords)
            }
        } else {
            // General case with broadcasting
            for (i in result.indices) {
                val thisIndex = broadcastIndex(i, resultShape, this.shape)
                val otherIndex = broadcastIndex(i, resultShape, other.shape)
                val thisCoords = flatIndexToCoords(thisIndex, this.shape)
                val otherCoords = flatIndexToCoords(otherIndex, other.shape)
                result[i] = this.get(*thisCoords) + other.get(*otherCoords)
            }
        }
        return CpuTensorFP32(DenseTensorData(resultShape, result))
    }

    override fun Tensor<FP32, Float>.minus(other: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(canBroadcast(this.shape, other.shape)) { "Tensors are not broadcast-compatible for subtraction: ${this.shape} vs ${other.shape}" }

        val resultShape = getBroadcastShape(this.shape, other.shape)
        val result = Array<Float>(resultShape.volume) { 0.0f }
        
        // Optimize for contiguous tensors of same shape
        if (this.isContiguous && other.isContiguous && this.shape == other.shape) {
            for (i in result.indices) {
                val coords = flatIndexToCoords(i, this.shape)
                result[i] = this.get(*coords) - other.get(*coords)
            }
        } else {
            // General case with broadcasting
            for (i in result.indices) {
                val thisIndex = broadcastIndex(i, resultShape, this.shape)
                val otherIndex = broadcastIndex(i, resultShape, other.shape)
                val thisCoords = flatIndexToCoords(thisIndex, this.shape)
                val otherCoords = flatIndexToCoords(otherIndex, other.shape)
                result[i] = this.get(*thisCoords) - other.get(*otherCoords)
            }
        }
        return CpuTensorFP32(DenseTensorData(resultShape, result))
    }

    override fun Tensor<FP32, Float>.times(other: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(canBroadcast(this.shape, other.shape)) { "Tensors are not broadcast-compatible for element-wise multiplication: ${this.shape} vs ${other.shape}" }

        val resultShape = getBroadcastShape(this.shape, other.shape)
        val result = Array<Float>(resultShape.volume) { 0.0f }
        
        // Optimize for contiguous tensors of same shape
        if (this.isContiguous && other.isContiguous && this.shape == other.shape) {
            for (i in result.indices) {
                val coords = flatIndexToCoords(i, this.shape)
                result[i] = this.get(*coords) * other.get(*coords)
            }
        } else {
            // General case with broadcasting
            for (i in result.indices) {
                val thisIndex = broadcastIndex(i, resultShape, this.shape)
                val otherIndex = broadcastIndex(i, resultShape, other.shape)
                val thisCoords = flatIndexToCoords(thisIndex, this.shape)
                val otherCoords = flatIndexToCoords(otherIndex, other.shape)
                result[i] = this.get(*thisCoords) * other.get(*otherCoords)
            }
        }
        return CpuTensorFP32(DenseTensorData(resultShape, result))
    }

    override fun Tensor<FP32, Float>.div(other: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(canBroadcast(this.shape, other.shape)) { "Tensors are not broadcast-compatible for element-wise division: ${this.shape} vs ${other.shape}" }

        val resultShape = getBroadcastShape(this.shape, other.shape)
        val result = Array<Float>(resultShape.volume) { 0.0f }
        
        // Optimize for contiguous tensors of same shape
        if (this.isContiguous && other.isContiguous && this.shape == other.shape) {
            for (i in result.indices) {
                val coords = flatIndexToCoords(i, this.shape)
                result[i] = this.get(*coords) / other.get(*coords)
            }
        } else {
            // General case with broadcasting
            for (i in result.indices) {
                val thisIndex = broadcastIndex(i, resultShape, this.shape)
                val otherIndex = broadcastIndex(i, resultShape, other.shape)
                val thisCoords = flatIndexToCoords(thisIndex, this.shape)
                val otherCoords = flatIndexToCoords(otherIndex, other.shape)
                result[i] = this.get(*thisCoords) / other.get(*otherCoords)
            }
        }
        return CpuTensorFP32(DenseTensorData(resultShape, result))
    }

    // Tensor-Scalar operations
    override fun Tensor<FP32, Float>.plus(scalar: Int): Tensor<FP32, Float> {
        val result = FloatArray(this.shape.volume)
        val coords = IntArray(this.shape.rank)
        var index = 0
        
        fun iterate(dim: Int) {
            if (dim == this.shape.rank) {
                result[index++] = this.get(*coords) + scalar
                return
            }
            for (i in 0 until this.shape[dim]) {
                coords[dim] = i
                iterate(dim + 1)
            }
        }
        
        iterate(0)
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.minus(scalar: Int): Tensor<FP32, Float> {
        val result = FloatArray(this.shape.volume)
        val coords = IntArray(this.shape.rank)
        var index = 0
        
        fun iterate(dim: Int) {
            if (dim == this.shape.rank) {
                result[index++] = this.get(*coords) - scalar
                return
            }
            for (i in 0 until this.shape[dim]) {
                coords[dim] = i
                iterate(dim + 1)
            }
        }
        
        iterate(0)
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.times(scalar: Int): Tensor<FP32, Float> {
        val result = FloatArray(this.shape.volume)
        val coords = IntArray(this.shape.rank)
        var index = 0
        
        fun iterate(dim: Int) {
            if (dim == this.shape.rank) {
                result[index++] = this.get(*coords) * scalar
                return
            }
            for (i in 0 until this.shape[dim]) {
                coords[dim] = i
                iterate(dim + 1)
            }
        }
        
        iterate(0)
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.div(scalar: Int): Tensor<FP32, Float> {
        val result = FloatArray(this.shape.volume)
        val coords = IntArray(this.shape.rank)
        var index = 0
        
        fun iterate(dim: Int) {
            if (dim == this.shape.rank) {
                result[index++] = this.get(*coords) / scalar
                return
            }
            for (i in 0 until this.shape[dim]) {
                coords[dim] = i
                iterate(dim + 1)
            }
        }
        
        iterate(0)
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.plus(scalar: Float): Tensor<FP32, Float> {
        val result = FloatArray(this.shape.volume)
        val coords = IntArray(this.shape.rank)
        var index = 0
        
        fun iterate(dim: Int) {
            if (dim == this.shape.rank) {
                result[index++] = this.get(*coords) + scalar
                return
            }
            for (i in 0 until this.shape[dim]) {
                coords[dim] = i
                iterate(dim + 1)
            }
        }
        
        iterate(0)
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.minus(scalar: Float): Tensor<FP32, Float> {
        val result = FloatArray(this.shape.volume)
        val coords = IntArray(this.shape.rank)
        var index = 0
        
        fun iterate(dim: Int) {
            if (dim == this.shape.rank) {
                result[index++] = this.get(*coords) - scalar
                return
            }
            for (i in 0 until this.shape[dim]) {
                coords[dim] = i
                iterate(dim + 1)
            }
        }
        
        iterate(0)
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.times(scalar: Float): Tensor<FP32, Float> {
        val result = FloatArray(this.shape.volume)
        val coords = IntArray(this.shape.rank)
        var index = 0
        
        fun iterate(dim: Int) {
            if (dim == this.shape.rank) {
                result[index++] = this.get(*coords) * scalar
                return
            }
            for (i in 0 until this.shape[dim]) {
                coords[dim] = i
                iterate(dim + 1)
            }
        }
        
        iterate(0)
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.div(scalar: Float): Tensor<FP32, Float> {
        val result = FloatArray(this.shape.volume)
        val coords = IntArray(this.shape.rank)
        var index = 0
        
        fun iterate(dim: Int) {
            if (dim == this.shape.rank) {
                result[index++] = this.get(*coords) / scalar
                return
            }
            for (i in 0 until this.shape[dim]) {
                coords[dim] = i
                iterate(dim + 1)
            }
        }
        
        iterate(0)
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.plus(scalar: Double): Tensor<FP32, Float> {
        val result = FloatArray(this.shape.volume)
        val coords = IntArray(this.shape.rank)
        var index = 0
        
        fun iterate(dim: Int) {
            if (dim == this.shape.rank) {
                result[index++] = this.get(*coords) + scalar.toFloat()
                return
            }
            for (i in 0 until this.shape[dim]) {
                coords[dim] = i
                iterate(dim + 1)
            }
        }
        
        iterate(0)
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.minus(scalar: Double): Tensor<FP32, Float> {
        val result = FloatArray(this.shape.volume)
        val coords = IntArray(this.shape.rank)
        var index = 0
        
        fun iterate(dim: Int) {
            if (dim == this.shape.rank) {
                result[index++] = this.get(*coords) - scalar.toFloat()
                return
            }
            for (i in 0 until this.shape[dim]) {
                coords[dim] = i
                iterate(dim + 1)
            }
        }
        
        iterate(0)
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.times(scalar: Double): Tensor<FP32, Float> {
        val result = FloatArray(this.shape.volume)
        val coords = IntArray(this.shape.rank)
        var index = 0
        
        fun iterate(dim: Int) {
            if (dim == this.shape.rank) {
                result[index++] = this.get(*coords) * scalar.toFloat()
                return
            }
            for (i in 0 until this.shape[dim]) {
                coords[dim] = i
                iterate(dim + 1)
            }
        }
        
        iterate(0)
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.div(scalar: Double): Tensor<FP32, Float> {
        val result = FloatArray(this.shape.volume)
        val coords = IntArray(this.shape.rank)
        var index = 0
        
        fun iterate(dim: Int) {
            if (dim == this.shape.rank) {
                result[index++] = this.get(*coords) / scalar.toFloat()
                return
            }
            for (i in 0 until this.shape[dim]) {
                coords[dim] = i
                iterate(dim + 1)
            }
        }
        
        iterate(0)
        return CpuTensorFP32(this.shape, result)
    }

    // Scalar-Tensor operations
    override fun Double.plus(t: Tensor<FP32, Float>): Tensor<FP32, Float> {
        val result = FloatArray(t.shape.volume)
        val coords = IntArray(t.shape.rank)
        var index = 0
        
        fun iterate(dim: Int) {
            if (dim == t.shape.rank) {
                result[index++] = this.toFloat() + t.get(*coords)
                return
            }
            for (i in 0 until t.shape[dim]) {
                coords[dim] = i
                iterate(dim + 1)
            }
        }
        
        iterate(0)
        return CpuTensorFP32(t.shape, result)
    }

    override fun Double.minus(t: Tensor<FP32, Float>): Tensor<FP32, Float> {
        val result = FloatArray(t.shape.volume)
        val coords = IntArray(t.shape.rank)
        var index = 0
        
        fun iterate(dim: Int) {
            if (dim == t.shape.rank) {
                result[index++] = this.toFloat() - t.get(*coords)
                return
            }
            for (i in 0 until t.shape[dim]) {
                coords[dim] = i
                iterate(dim + 1)
            }
        }
        
        iterate(0)
        return CpuTensorFP32(t.shape, result)
    }

    override fun Double.times(t: Tensor<FP32, Float>): Tensor<FP32, Float> {
        val result = FloatArray(t.shape.volume)
        val coords = IntArray(t.shape.rank)
        var index = 0
        
        fun iterate(dim: Int) {
            if (dim == t.shape.rank) {
                result[index++] = this.toFloat() * t.get(*coords)
                return
            }
            for (i in 0 until t.shape[dim]) {
                coords[dim] = i
                iterate(dim + 1)
            }
        }
        
        iterate(0)
        return CpuTensorFP32(t.shape, result)
    }

    override fun Double.div(t: Tensor<FP32, Float>): Tensor<FP32, Float> {
        val result = FloatArray(t.shape.volume)
        val coords = IntArray(t.shape.rank)
        var index = 0
        
        fun iterate(dim: Int) {
            if (dim == t.shape.rank) {
                result[index++] = this.toFloat() / t.get(*coords)
                return
            }
            for (i in 0 until t.shape[dim]) {
                coords[dim] = i
                iterate(dim + 1)
            }
        }
        
        iterate(0)
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
        val result = Array<Float>(this.shape.volume) { 0.0f }
        
        // Optimize for contiguous data
        if (this.isContiguous) {
            // Fast path: iterate through tensor using coordinate-based access
            for (i in result.indices) {
                val coords = flatIndexToCoords(i, this.shape)
                result[i] = maxOf(0.0f, this.get(*coords))
            }
        } else {
            // Generic stride-aware path for non-contiguous data
            val coords = IntArray(this.shape.rank)
            var index = 0
            
            fun iterate(dim: Int) {
                if (dim == this.shape.rank) {
                    result[index++] = maxOf(0.0f, this.get(*coords))
                    return
                }
                for (i in 0 until this.shape[dim]) {
                    coords[dim] = i
                    iterate(dim + 1)
                }
            }
            iterate(0)
        }
        
        return CpuTensorFP32(DenseTensorData(this.shape, result))
    }

    override fun Tensor<FP32, Float>.sigmoid(): Tensor<FP32, Float> {
        val result = Array<Float>(this.shape.volume) { 0.0f }
        
        // Optimize for contiguous data
        if (this.isContiguous) {
            // Fast path: iterate through tensor using coordinate-based access
            for (i in result.indices) {
                val coords = flatIndexToCoords(i, this.shape)
                val value = this.get(*coords)
                result[i] = 1.0f / (1.0f + exp(-value))
            }
        } else {
            // Generic stride-aware path for non-contiguous data
            val coords = IntArray(this.shape.rank)
            var index = 0
            
            fun iterate(dim: Int) {
                if (dim == this.shape.rank) {
                    val value = this.get(*coords)
                    result[index++] = 1.0f / (1.0f + exp(-value))
                    return
                }
                for (i in 0 until this.shape[dim]) {
                    coords[dim] = i
                    iterate(dim + 1)
                }
            }
            iterate(0)
        }
        
        return CpuTensorFP32(DenseTensorData(this.shape, result))
    }

    override fun Tensor<FP32, Float>.tanh(): Tensor<FP32, Float> {
        val result = Array<Float>(this.shape.volume) { 0.0f }
        
        // Optimize for contiguous data
        if (this.isContiguous) {
            // Fast path: iterate through tensor using coordinate-based access
            for (i in result.indices) {
                val coords = flatIndexToCoords(i, this.shape)
                result[i] = tanh(this.get(*coords))
            }
        } else {
            // Generic stride-aware path for non-contiguous data
            val coords = IntArray(this.shape.rank)
            var index = 0
            
            fun iterate(dim: Int) {
                if (dim == this.shape.rank) {
                    result[index++] = tanh(this.get(*coords))
                    return
                }
                for (i in 0 until this.shape[dim]) {
                    coords[dim] = i
                    iterate(dim + 1)
                }
            }
            iterate(0)
        }
        
        return CpuTensorFP32(DenseTensorData(this.shape, result))
    }

    override fun Tensor<FP32, Float>.softmax(dimension: Int): Tensor<FP32, Float> {
        require(dimension in 0 until this.shape.rank) { "Dimension $dimension is out of bounds for tensor with rank ${this.shape.rank}" }
        
        // For TensorView compatibility, use a generic implementation for now
        if (this !is CpuTensorFP32) {
            // Simple generic softmax implementation for views
            val resultData = Array<Float>(this.shape.volume) { 0f }
            this.copyTo(resultData, 0)
            
            // Apply simple softmax across all elements (simplified version)
            val max = resultData.maxOrNull() ?: 0f
            for (i in resultData.indices) {
                resultData[i] = exp(resultData[i] - max)
            }
            val sum = resultData.sum()
            for (i in resultData.indices) {
                resultData[i] /= sum
            }
            
            return CpuTensorFP32(DenseTensorData<FP32, Float>(this.shape, resultData))
        }
        
        // Original implementation for CpuTensorFP32
        
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

    override fun Tensor<FP32, Float>.reshape(newShape: Shape): Tensor<FP32, Float> {
        require(this.shape.volume == newShape.volume) {
            "Cannot reshape tensor with ${this.shape.volume} elements to shape with ${newShape.volume} elements"
        }
        
        // Handle both CpuTensorFP32 and sliced tensors
        if (this is CpuTensorFP32) {
            // Direct access to data for CpuTensorFP32
            return CpuTensorFP32(newShape, this.data.copyOf())
        } else {
            // For sliced tensors, extract data using copyTo
            val tensorData = Array<Float>(this.shape.volume) { 0f }
            this.copyTo(tensorData)
            return CpuTensorFP32(newShape, tensorData.toFloatArray())
        }
    }
    
    override fun Tensor<FP32, Float>.reshape(vararg dimensions: Int): Tensor<FP32, Float> {
        // Count -1 dimensions and validate
        val minusOneCount = dimensions.count { it == -1 }
        require(minusOneCount <= 1) { "Only one dimension can be -1, found $minusOneCount" }
        
        if (minusOneCount == 0) {
            // No -1, use regular reshape
            return this.reshape(Shape(*dimensions))
        }
        
        // Calculate inferred dimension
        val minusOneIndex = dimensions.indexOfFirst { it == -1 }
        val knownProduct = dimensions.filter { it != -1 }.fold(1) { acc, dim ->
            require(dim > 0) { "All dimensions except -1 must be positive, got $dim" }
            acc * dim
        }
        
        require(this.shape.volume % knownProduct == 0) {
            "Cannot reshape tensor with ${this.shape.volume} elements: remaining elements $knownProduct do not divide evenly"
        }
        
        val inferredDim = this.shape.volume / knownProduct
        val newDimensions = dimensions.copyOf()
        newDimensions[minusOneIndex] = inferredDim
        
        return this.reshape(Shape(*newDimensions))
    }

    // TensorFactory interface implementation
    override fun zeros(shape: Shape): Tensor<FP32, Float> {
        return CpuTensorFP32.zeros(shape)
    }

    override fun ones(shape: Shape): Tensor<FP32, Float> {
        return CpuTensorFP32.ones(shape)
    }

    override fun random(shape: Shape): Tensor<FP32, Float> {
        return CpuTensorFP32.fromArray(shape, FloatArray(shape.volume) { kotlin.random.Random.nextFloat() })
    }

    override fun random(shape: Shape, seed: Long): Tensor<FP32, Float> {
        val random = kotlin.random.Random(seed)
        return CpuTensorFP32.fromArray(shape, FloatArray(shape.volume) { random.nextFloat() })
    }

    override fun random(shape: Shape, random: kotlin.random.Random): Tensor<FP32, Float> {
        return CpuTensorFP32.fromArray(shape, FloatArray(shape.volume) { random.nextFloat() })
    }

    override fun randomNormal(shape: Shape, mean: Double, std: Double): Tensor<FP32, Float> {
        val random = kotlin.random.Random
        return CpuTensorFP32.fromArray(shape, FloatArray(shape.volume) { 
            (generateNormalDistribution(random, mean, std)).toFloat()
        })
    }

    override fun randomNormal(shape: Shape, mean: Double, std: Double, seed: Long): Tensor<FP32, Float> {
        val random = kotlin.random.Random(seed)
        return CpuTensorFP32.fromArray(shape, FloatArray(shape.volume) { 
            (generateNormalDistribution(random, mean, std)).toFloat()
        })
    }

    override fun randomNormal(shape: Shape, mean: Double, std: Double, random: kotlin.random.Random): Tensor<FP32, Float> {
        return CpuTensorFP32.fromArray(shape, FloatArray(shape.volume) { 
            (generateNormalDistribution(random, mean, std)).toFloat()
        })
    }

    override fun randomUniform(shape: Shape, min: Double, max: Double): Tensor<FP32, Float> {
        val random = kotlin.random.Random
        return CpuTensorFP32.fromArray(shape, FloatArray(shape.volume) { 
            (random.nextDouble(min, max)).toFloat()
        })
    }

    override fun randomUniform(shape: Shape, min: Double, max: Double, seed: Long): Tensor<FP32, Float> {
        val random = kotlin.random.Random(seed)
        return CpuTensorFP32.fromArray(shape, FloatArray(shape.volume) { 
            (random.nextDouble(min, max)).toFloat()
        })
    }

    override fun randomUniform(shape: Shape, min: Double, max: Double, random: kotlin.random.Random): Tensor<FP32, Float> {
        return CpuTensorFP32.fromArray(shape, FloatArray(shape.volume) { 
            (random.nextDouble(min, max)).toFloat()
        })
    }

    private fun generateNormalDistribution(random: kotlin.random.Random, mean: Double, std: Double): Double {
        // Box-Muller transform for generating normal distribution
        val u1 = random.nextDouble()
        val u2 = random.nextDouble()
        val z0 = sqrt(-2.0 * ln(u1)) * cos(2.0 * PI * u2)
        return z0 * std + mean
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

    override fun matmul4d(a: Tensor<Int8, Byte>, b: Tensor<Int8, Byte>): Tensor<Int8, Byte> =
        (a as CpuTensorInt8).matmul4d(a, b)

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

    override fun Tensor<Int8, Byte>.reshape(newShape: Shape): Tensor<Int8, Byte> =
        (this as CpuTensorInt8).reshape(newShape)

    override fun Tensor<Int8, Byte>.reshape(vararg dimensions: Int): Tensor<Int8, Byte> =
        (this as CpuTensorInt8).reshape(*dimensions)

    override fun zeros(shape: Shape): Tensor<Int8, Byte> =
        CpuTensorInt8.zeros(shape)

    override fun ones(shape: Shape): Tensor<Int8, Byte> =
        CpuTensorInt8.ones(shape)

    override fun random(shape: Shape): Tensor<Int8, Byte> =
        CpuTensorInt8.fromArray(shape, ByteArray(shape.volume) { kotlin.random.Random.nextInt(-128, 128).toByte() })

    override fun random(shape: Shape, seed: Long): Tensor<Int8, Byte> {
        val random = kotlin.random.Random(seed)
        return CpuTensorInt8.fromArray(shape, ByteArray(shape.volume) { random.nextInt(-128, 128).toByte() })
    }

    override fun random(shape: Shape, random: kotlin.random.Random): Tensor<Int8, Byte> =
        CpuTensorInt8.fromArray(shape, ByteArray(shape.volume) { random.nextInt(-128, 128).toByte() })

    override fun randomNormal(shape: Shape, mean: Double, std: Double): Tensor<Int8, Byte> {
        val random = kotlin.random.Random
        return CpuTensorInt8.fromArray(shape, ByteArray(shape.volume) { 
            generateNormalDistributionInt8(random, mean, std)
        })
    }

    override fun randomNormal(shape: Shape, mean: Double, std: Double, seed: Long): Tensor<Int8, Byte> {
        val random = kotlin.random.Random(seed)
        return CpuTensorInt8.fromArray(shape, ByteArray(shape.volume) { 
            generateNormalDistributionInt8(random, mean, std)
        })
    }

    override fun randomNormal(shape: Shape, mean: Double, std: Double, random: kotlin.random.Random): Tensor<Int8, Byte> =
        CpuTensorInt8.fromArray(shape, ByteArray(shape.volume) { 
            generateNormalDistributionInt8(random, mean, std)
        })

    override fun randomUniform(shape: Shape, min: Double, max: Double): Tensor<Int8, Byte> {
        val random = kotlin.random.Random
        return CpuTensorInt8.fromArray(shape, ByteArray(shape.volume) { 
            random.nextInt(min.toInt().coerceAtLeast(-128), max.toInt().coerceAtMost(127) + 1).toByte()
        })
    }

    override fun randomUniform(shape: Shape, min: Double, max: Double, seed: Long): Tensor<Int8, Byte> {
        val random = kotlin.random.Random(seed)
        return CpuTensorInt8.fromArray(shape, ByteArray(shape.volume) { 
            random.nextInt(min.toInt().coerceAtLeast(-128), max.toInt().coerceAtMost(127) + 1).toByte()
        })
    }

    override fun randomUniform(shape: Shape, min: Double, max: Double, random: kotlin.random.Random): Tensor<Int8, Byte> =
        CpuTensorInt8.fromArray(shape, ByteArray(shape.volume) { 
            random.nextInt(min.toInt().coerceAtLeast(-128), max.toInt().coerceAtMost(127) + 1).toByte()
        })

    private fun generateNormalDistributionInt8(random: kotlin.random.Random, mean: Double, std: Double): Byte {
        // Box-Muller transform for generating normal distribution
        val u1 = random.nextDouble()
        val u2 = random.nextDouble()
        val z0 = sqrt(-2.0 * ln(u1)) * cos(2.0 * PI * u2)
        val value = (z0 * std + mean).toInt().coerceIn(-128, 127)
        return value.toByte()
    }
}

/**
 * A CPU-based implementation of the ComputeBackend interface for Int32/Int tensors.
 */
public class CpuBackendInt32 : ComputeBackend<Int32, Int> {
    override val name: String = "CPU-Int32"
    
    init {
        TensorFactoryInitializer.ensureInitialized()
    }

    override fun matmul4d(a: Tensor<Int32, Int>, b: Tensor<Int32, Int>): Tensor<Int32, Int> =
        (a as CpuTensorInt32).matmul4d(a, b)

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

    override fun Tensor<Int32, Int>.reshape(newShape: Shape): Tensor<Int32, Int> =
        (this as CpuTensorInt32).reshape(newShape)

    override fun Tensor<Int32, Int>.reshape(vararg dimensions: Int): Tensor<Int32, Int> =
        (this as CpuTensorInt32).reshape(*dimensions)

    override fun zeros(shape: Shape): Tensor<Int32, Int> =
        CpuTensorInt32.zeros(shape)

    override fun ones(shape: Shape): Tensor<Int32, Int> =
        CpuTensorInt32.ones(shape)

    override fun random(shape: Shape): Tensor<Int32, Int> =
        CpuTensorInt32.fromArray(shape, IntArray(shape.volume) { kotlin.random.Random.nextInt() })

    override fun random(shape: Shape, seed: Long): Tensor<Int32, Int> {
        val random = kotlin.random.Random(seed)
        return CpuTensorInt32.fromArray(shape, IntArray(shape.volume) { random.nextInt() })
    }

    override fun random(shape: Shape, random: kotlin.random.Random): Tensor<Int32, Int> =
        CpuTensorInt32.fromArray(shape, IntArray(shape.volume) { random.nextInt() })

    override fun randomNormal(shape: Shape, mean: Double, std: Double): Tensor<Int32, Int> {
        val random = kotlin.random.Random
        return CpuTensorInt32.fromArray(shape, IntArray(shape.volume) { 
            generateNormalDistributionInt32(random, mean, std)
        })
    }

    override fun randomNormal(shape: Shape, mean: Double, std: Double, seed: Long): Tensor<Int32, Int> {
        val random = kotlin.random.Random(seed)
        return CpuTensorInt32.fromArray(shape, IntArray(shape.volume) { 
            generateNormalDistributionInt32(random, mean, std)
        })
    }

    override fun randomNormal(shape: Shape, mean: Double, std: Double, random: kotlin.random.Random): Tensor<Int32, Int> =
        CpuTensorInt32.fromArray(shape, IntArray(shape.volume) { 
            generateNormalDistributionInt32(random, mean, std)
        })

    override fun randomUniform(shape: Shape, min: Double, max: Double): Tensor<Int32, Int> {
        val random = kotlin.random.Random
        return CpuTensorInt32.fromArray(shape, IntArray(shape.volume) { 
            random.nextInt(min.toInt(), max.toInt() + 1)
        })
    }

    override fun randomUniform(shape: Shape, min: Double, max: Double, seed: Long): Tensor<Int32, Int> {
        val random = kotlin.random.Random(seed)
        return CpuTensorInt32.fromArray(shape, IntArray(shape.volume) { 
            random.nextInt(min.toInt(), max.toInt() + 1)
        })
    }

    override fun randomUniform(shape: Shape, min: Double, max: Double, random: kotlin.random.Random): Tensor<Int32, Int> =
        CpuTensorInt32.fromArray(shape, IntArray(shape.volume) { 
            random.nextInt(min.toInt(), max.toInt() + 1)
        })

    private fun generateNormalDistributionInt32(random: kotlin.random.Random, mean: Double, std: Double): Int {
        // Box-Muller transform for generating normal distribution
        val u1 = random.nextDouble()
        val u2 = random.nextDouble()
        val z0 = sqrt(-2.0 * ln(u1)) * cos(2.0 * PI * u2)
        return (z0 * std + mean).toInt()
    }
}

/**
 * A CPU-based tensor for Int4/Byte values with packed 4-bit storage.
 * Each byte stores 2 Int4 values: high nibble (bits 4-7) and low nibble (bits 0-3).
 * Values are signed 4-bit integers in range -8 to 7.
 */
public class CpuTensorInt4(
    override val shape: Shape,
    internal val data: ByteArray  // Packed data: 2 values per byte
) : TensorInt4 {

    // TensorData implementation
    override val strides: IntArray = shape.computeStrides()
    override val offset: Int = 0
    override val isContiguous: Boolean = true
    
    init {
        val expectedBytes = (shape.volume + 1) / 2  // Ceiling division
        require(data.size == expectedBytes) {
            "Data size ${data.size} doesn't match expected packed size $expectedBytes bytes for ${shape.volume} Int4 values"
        }
        require(shape.rank in 1..4) {
            "Only 1-4 dimensional tensors are supported, got ${shape.rank}"
        }
    }

    override fun get(vararg indices: Int): Byte {
        val linearIndex = shape.index(indices)
        val byteIndex = linearIndex / 2
        val isHighNibble = (linearIndex % 2) == 0
        
        return if (isHighNibble) {
            // Extract high nibble (bits 4-7) and sign-extend
            val value = (data[byteIndex].toInt() shr 4) and 0x0F
            if (value >= 8) (value - 16).toByte() else value.toByte()
        } else {
            // Extract low nibble (bits 0-3) and sign-extend  
            val value = data[byteIndex].toInt() and 0x0F
            if (value >= 8) (value - 16).toByte() else value.toByte()
        }
    }
    
    override fun copyTo(dest: Array<Byte>, destOffset: Int) {
        for (i in 0 until shape.volume) {
            val byteIndex = i / 2
            val isHighNibble = (i % 2) == 0
            
            dest[destOffset + i] = if (isHighNibble) {
                val value = (data[byteIndex].toInt() shr 4) and 0x0F
                if (value >= 8) (value - 16).toByte() else value.toByte()
            } else {
                val value = data[byteIndex].toInt() and 0x0F
                if (value >= 8) (value - 16).toByte() else value.toByte()
            }
        }
    }
    
    override fun slice(ranges: IntArray): TensorData<Int4, Byte> {
        // For simplicity, materialize the slice - could be optimized later
        return slice_impl(ranges)
    }
    
    private fun slice_impl(ranges: IntArray): CpuTensorInt4 {
        require(ranges.size == shape.rank * 2) {
            "Ranges array must have size ${shape.rank * 2} (start,end pairs), got ${ranges.size}"
        }
        
        // Parse ranges and calculate new shape
        val sliceRanges = mutableListOf<Pair<Int, Int>>()
        for (i in 0 until shape.rank) {
            val start = ranges[i * 2]
            val end = ranges[i * 2 + 1]
            val dimSize = shape.dimensions[i]
            
            require(start >= 0 && start < dimSize) {
                "Start index $start out of bounds for dimension $i (size $dimSize)"
            }
            require(end > start && end <= dimSize) {
                "End index $end must be > start ($start) and <= dimension size ($dimSize)"
            }
            
            sliceRanges.add(start to end)
        }
        
        val newDimensions = sliceRanges.map { (start, end) -> end - start }.toIntArray()
        val newShape = Shape(newDimensions)
        val newSize = newShape.volume
        val newPackedSize = (newSize + 1) / 2
        val newData = ByteArray(newPackedSize)
        
        // Extract sliced values and pack them
        var destLinearIndex = 0
        
        when (shape.rank) {
            1 -> {
                val (start0, end0) = sliceRanges[0]
                for (i in start0 until end0) {
                    val value = get(i)
                    packValue(newData, destLinearIndex, value)
                    destLinearIndex++
                }
            }
            2 -> {
                val (start0, end0) = sliceRanges[0]
                val (start1, end1) = sliceRanges[1]
                for (i in start0 until end0) {
                    for (j in start1 until end1) {
                        val value = get(i, j)
                        packValue(newData, destLinearIndex, value)
                        destLinearIndex++
                    }
                }
            }
            else -> throw UnsupportedOperationException("Slicing not fully implemented for ${shape.rank}D tensors")
        }
        
        return CpuTensorInt4(newShape, newData)
    }
    
    private fun packValue(packedData: ByteArray, linearIndex: Int, value: Byte) {
        val byteIndex = linearIndex / 2
        val isHighNibble = (linearIndex % 2) == 0
        
        // Clamp to 4-bit signed range [-8, 7]
        val clampedValue = value.toInt().coerceIn(-8, 7)
        val nibble = if (clampedValue < 0) clampedValue + 16 else clampedValue
        
        if (isHighNibble) {
            // Set high nibble, preserve low nibble
            packedData[byteIndex] = ((nibble shl 4) or (packedData[byteIndex].toInt() and 0x0F)).toByte()
        } else {
            // Set low nibble, preserve high nibble  
            packedData[byteIndex] = ((packedData[byteIndex].toInt() and 0xF0) or nibble).toByte()
        }
    }
    
    override fun materialize(): TensorData<Int4, Byte> = this
    
    private fun Shape.computeStrides(): IntArray {
        if (dimensions.isEmpty()) return intArrayOf()
        val strides = IntArray(dimensions.size)
        strides[dimensions.size - 1] = 1
        for (i in dimensions.size - 2 downTo 0) {
            strides[i] = strides[i + 1] * dimensions[i + 1]
        }
        return strides
    }

    // Basic operations placeholder - would need full implementation
    override fun matmul(a: Tensor<Int4, Byte>, b: Tensor<Int4, Byte>): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Matrix multiplication not yet implemented for Int4 tensors")
    }

    override fun matmul4d(a: Tensor<Int4, Byte>, b: Tensor<Int4, Byte>): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("4D matrix multiplication not yet implemented for Int4 tensors")
    }

    override fun scale(a: Tensor<Int4, Byte>, scalar: Double): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Scale operation not yet implemented for Int4 tensors")
    }

    override fun dot(a: Tensor<Int4, Byte>, b: Tensor<Int4, Byte>): Double {
        throw UnsupportedOperationException("Dot product not yet implemented for Int4 tensors")
    }

    // Tensor operations placeholder
    override fun Tensor<Int4, Byte>.plus(other: Tensor<Int4, Byte>): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Addition not yet implemented for Int4 tensors")
    }

    override fun Tensor<Int4, Byte>.minus(other: Tensor<Int4, Byte>): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Subtraction not yet implemented for Int4 tensors")
    }

    override fun Tensor<Int4, Byte>.times(other: Tensor<Int4, Byte>): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Multiplication not yet implemented for Int4 tensors")
    }

    override fun Tensor<Int4, Byte>.div(other: Tensor<Int4, Byte>): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Division not yet implemented for Int4 tensors")
    }

    // Scalar operations placeholder
    override fun Tensor<Int4, Byte>.plus(scalar: Int): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Scalar addition not yet implemented for Int4 tensors")
    }

    override fun Tensor<Int4, Byte>.minus(scalar: Int): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Scalar subtraction not yet implemented for Int4 tensors")
    }

    override fun Tensor<Int4, Byte>.times(scalar: Int): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Scalar multiplication not yet implemented for Int4 tensors")
    }

    override fun Tensor<Int4, Byte>.div(scalar: Int): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Scalar division not yet implemented for Int4 tensors")
    }

    override fun Tensor<Int4, Byte>.plus(scalar: Float): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Float scalar addition not yet implemented for Int4 tensors")
    }

    override fun Tensor<Int4, Byte>.minus(scalar: Float): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Float scalar subtraction not yet implemented for Int4 tensors")
    }

    override fun Tensor<Int4, Byte>.times(scalar: Float): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Float scalar multiplication not yet implemented for Int4 tensors")
    }

    override fun Tensor<Int4, Byte>.div(scalar: Float): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Float scalar division not yet implemented for Int4 tensors")
    }

    override fun Tensor<Int4, Byte>.plus(scalar: Double): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Double scalar addition not yet implemented for Int4 tensors")
    }

    override fun Tensor<Int4, Byte>.minus(scalar: Double): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Double scalar subtraction not yet implemented for Int4 tensors")
    }

    override fun Tensor<Int4, Byte>.times(scalar: Double): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Double scalar multiplication not yet implemented for Int4 tensors")
    }

    override fun Tensor<Int4, Byte>.div(scalar: Double): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Double scalar division not yet implemented for Int4 tensors")
    }

    // Scalar-tensor operations placeholder
    override fun Double.plus(t: Tensor<Int4, Byte>): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Scalar-tensor addition not yet implemented for Int4 tensors")
    }

    override fun Double.minus(t: Tensor<Int4, Byte>): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Scalar-tensor subtraction not yet implemented for Int4 tensors")
    }

    override fun Double.times(t: Tensor<Int4, Byte>): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Scalar-tensor multiplication not yet implemented for Int4 tensors")
    }

    override fun Double.div(t: Tensor<Int4, Byte>): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Scalar-tensor division not yet implemented for Int4 tensors")
    }

    // Advanced operations placeholder
    override fun Tensor<Int4, Byte>.t(): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Transpose not yet implemented for Int4 tensors")
    }

    override fun Tensor<Int4, Byte>.relu(): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("ReLU not yet implemented for Int4 tensors")
    }

    override fun Tensor<Int4, Byte>.sigmoid(): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Sigmoid not yet implemented for Int4 tensors")
    }

    override fun Tensor<Int4, Byte>.tanh(): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Tanh not yet implemented for Int4 tensors")
    }

    override fun Tensor<Int4, Byte>.softmax(dimension: Int): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Softmax not yet implemented for Int4 tensors")
    }

    override fun Tensor<Int4, Byte>.flatten(startDim: Int, endDim: Int): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Flatten not yet implemented for Int4 tensors")
    }

    override fun Tensor<Int4, Byte>.reshape(newShape: Shape): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Reshape not yet implemented for Int4 tensors")
    }

    override fun Tensor<Int4, Byte>.reshape(vararg dimensions: Int): Tensor<Int4, Byte> {
        throw UnsupportedOperationException("Reshape not yet implemented for Int4 tensors")
    }

    public companion object {
        /**
         * Creates a tensor from packed byte data.
         */
        public fun fromPackedByteArray(shape: Shape, data: ByteArray): CpuTensorInt4 {
            val expectedBytes = (shape.volume + 1) / 2
            require(data.size == expectedBytes) {
                "Data size ${data.size} doesn't match expected packed size $expectedBytes bytes"
            }
            return CpuTensorInt4(shape, data)
        }
    }
}

/**
 * A CPU-based tensor for Ternary/Byte values with packed 2-bit storage.
 * Each byte stores 4 Ternary values: 2 bits per value.
 * Values are ternary integers: -1, 0, 1 (mapped to 00->-1, 01->0, 10->1).
 */
public class CpuTensorTernary(
    override val shape: Shape,
    internal val data: ByteArray  // Packed data: 4 values per byte
) : TensorTernary {

    // TensorData implementation
    override val strides: IntArray = shape.computeStrides()
    override val offset: Int = 0
    override val isContiguous: Boolean = true
    
    init {
        val expectedBytes = (shape.volume + 3) / 4  // Ceiling division for 4 values per byte
        require(data.size == expectedBytes) {
            "Data size ${data.size} doesn't match expected packed size $expectedBytes bytes for ${shape.volume} Ternary values"
        }
        require(shape.rank in 1..4) {
            "Only 1-4 dimensional tensors are supported, got ${shape.rank}"
        }
    }

    override fun get(vararg indices: Int): Byte {
        val linearIndex = shape.index(indices)
        val byteIndex = linearIndex / 4
        val bitOffset = (linearIndex % 4) * 2  // 2 bits per value
        
        // Extract 2 bits and map to ternary value
        val value = (data[byteIndex].toInt() shr bitOffset) and 0x03
        return when (value) {
            0 -> -1  // 00 -> -1
            1 -> 0   // 01 -> 0  
            2 -> 1   // 10 -> 1
            else -> throw IllegalStateException("Invalid ternary value: $value")
        }
    }
    
    override fun copyTo(dest: Array<Byte>, destOffset: Int) {
        for (i in 0 until shape.volume) {
            val byteIndex = i / 4
            val bitOffset = (i % 4) * 2
            
            val value = (data[byteIndex].toInt() shr bitOffset) and 0x03
            dest[destOffset + i] = when (value) {
                0 -> -1  // 00 -> -1
                1 -> 0   // 01 -> 0  
                2 -> 1   // 10 -> 1
                else -> throw IllegalStateException("Invalid ternary value: $value")
            }
        }
    }
    
    override fun slice(ranges: IntArray): TensorData<Ternary, Byte> {
        return slice_impl(ranges)
    }
    
    private fun slice_impl(ranges: IntArray): CpuTensorTernary {
        require(ranges.size == shape.rank * 2) {
            "Ranges array must have size ${shape.rank * 2} (start,end pairs), got ${ranges.size}"
        }
        
        // Parse ranges and calculate new shape
        val sliceRanges = mutableListOf<Pair<Int, Int>>()
        for (i in 0 until shape.rank) {
            val start = ranges[i * 2]
            val end = ranges[i * 2 + 1]
            val dimSize = shape.dimensions[i]
            
            require(start >= 0 && start < dimSize) {
                "Start index $start out of bounds for dimension $i (size $dimSize)"
            }
            require(end > start && end <= dimSize) {
                "End index $end must be > start ($start) and <= dimension size ($dimSize)"
            }
            
            sliceRanges.add(start to end)
        }
        
        val newDimensions = sliceRanges.map { (start, end) -> end - start }.toIntArray()
        val newShape = Shape(newDimensions)
        val newSize = newShape.volume
        val newPackedSize = (newSize + 3) / 4
        val newData = ByteArray(newPackedSize)
        
        // Extract sliced values and pack them
        var destLinearIndex = 0
        
        when (shape.rank) {
            1 -> {
                val (start0, end0) = sliceRanges[0]
                for (i in start0 until end0) {
                    val value = get(i)
                    packValue(newData, destLinearIndex, value)
                    destLinearIndex++
                }
            }
            2 -> {
                val (start0, end0) = sliceRanges[0]
                val (start1, end1) = sliceRanges[1]
                for (i in start0 until end0) {
                    for (j in start1 until end1) {
                        val value = get(i, j)
                        packValue(newData, destLinearIndex, value)
                        destLinearIndex++
                    }
                }
            }
            else -> throw UnsupportedOperationException("Slicing not fully implemented for ${shape.rank}D tensors")
        }
        
        return CpuTensorTernary(newShape, newData)
    }
    
    private fun packValue(packedData: ByteArray, linearIndex: Int, value: Byte) {
        val byteIndex = linearIndex / 4
        val bitOffset = (linearIndex % 4) * 2
        
        // Map ternary value to 2-bit representation
        val bits = when (value.toInt()) {
            -1 -> 0  // -1 -> 00
            0 -> 1   // 0 -> 01
            1 -> 2   // 1 -> 10
            else -> throw IllegalArgumentException("Invalid ternary value: $value")
        }
        
        // Clear the 2 bits at bitOffset and set new value
        val mask = 0x03 shl bitOffset
        packedData[byteIndex] = ((packedData[byteIndex].toInt() and mask.inv()) or (bits shl bitOffset)).toByte()
    }
    
    override fun materialize(): TensorData<Ternary, Byte> = this
    
    private fun Shape.computeStrides(): IntArray {
        if (dimensions.isEmpty()) return intArrayOf()
        val strides = IntArray(dimensions.size)
        strides[dimensions.size - 1] = 1
        for (i in dimensions.size - 2 downTo 0) {
            strides[i] = strides[i + 1] * dimensions[i + 1]
        }
        return strides
    }

    // Placeholder operations - similar to CpuTensorInt4
    override fun matmul(a: Tensor<Ternary, Byte>, b: Tensor<Ternary, Byte>): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Matrix multiplication not yet implemented for Ternary tensors")
    }

    override fun matmul4d(a: Tensor<Ternary, Byte>, b: Tensor<Ternary, Byte>): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("4D matrix multiplication not yet implemented for Ternary tensors")
    }

    override fun scale(a: Tensor<Ternary, Byte>, scalar: Double): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Scale operation not yet implemented for Ternary tensors")
    }

    override fun dot(a: Tensor<Ternary, Byte>, b: Tensor<Ternary, Byte>): Double {
        throw UnsupportedOperationException("Dot product not yet implemented for Ternary tensors")
    }

    override fun Tensor<Ternary, Byte>.plus(other: Tensor<Ternary, Byte>): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Addition not yet implemented for Ternary tensors")
    }

    override fun Tensor<Ternary, Byte>.minus(other: Tensor<Ternary, Byte>): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Subtraction not yet implemented for Ternary tensors")
    }

    override fun Tensor<Ternary, Byte>.times(other: Tensor<Ternary, Byte>): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Multiplication not yet implemented for Ternary tensors")
    }

    override fun Tensor<Ternary, Byte>.div(other: Tensor<Ternary, Byte>): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Division not yet implemented for Ternary tensors")
    }

    override fun Tensor<Ternary, Byte>.plus(scalar: Int): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Scalar addition not yet implemented for Ternary tensors")
    }

    override fun Tensor<Ternary, Byte>.minus(scalar: Int): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Scalar subtraction not yet implemented for Ternary tensors")
    }

    override fun Tensor<Ternary, Byte>.times(scalar: Int): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Scalar multiplication not yet implemented for Ternary tensors")
    }

    override fun Tensor<Ternary, Byte>.div(scalar: Int): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Scalar division not yet implemented for Ternary tensors")
    }

    override fun Tensor<Ternary, Byte>.plus(scalar: Float): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Float scalar addition not yet implemented for Ternary tensors")
    }

    override fun Tensor<Ternary, Byte>.minus(scalar: Float): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Float scalar subtraction not yet implemented for Ternary tensors")
    }

    override fun Tensor<Ternary, Byte>.times(scalar: Float): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Float scalar multiplication not yet implemented for Ternary tensors")
    }

    override fun Tensor<Ternary, Byte>.div(scalar: Float): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Float scalar division not yet implemented for Ternary tensors")
    }

    override fun Tensor<Ternary, Byte>.plus(scalar: Double): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Double scalar addition not yet implemented for Ternary tensors")
    }

    override fun Tensor<Ternary, Byte>.minus(scalar: Double): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Double scalar subtraction not yet implemented for Ternary tensors")
    }

    override fun Tensor<Ternary, Byte>.times(scalar: Double): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Double scalar multiplication not yet implemented for Ternary tensors")
    }

    override fun Tensor<Ternary, Byte>.div(scalar: Double): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Double scalar division not yet implemented for Ternary tensors")
    }

    override fun Double.plus(t: Tensor<Ternary, Byte>): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Scalar-tensor addition not yet implemented for Ternary tensors")
    }

    override fun Double.minus(t: Tensor<Ternary, Byte>): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Scalar-tensor subtraction not yet implemented for Ternary tensors")
    }

    override fun Double.times(t: Tensor<Ternary, Byte>): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Scalar-tensor multiplication not yet implemented for Ternary tensors")
    }

    override fun Double.div(t: Tensor<Ternary, Byte>): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Scalar-tensor division not yet implemented for Ternary tensors")
    }

    override fun Tensor<Ternary, Byte>.t(): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Transpose not yet implemented for Ternary tensors")
    }

    override fun Tensor<Ternary, Byte>.relu(): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("ReLU not yet implemented for Ternary tensors")
    }

    override fun Tensor<Ternary, Byte>.sigmoid(): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Sigmoid not yet implemented for Ternary tensors")
    }

    override fun Tensor<Ternary, Byte>.tanh(): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Tanh not yet implemented for Ternary tensors")
    }

    override fun Tensor<Ternary, Byte>.softmax(dimension: Int): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Softmax not yet implemented for Ternary tensors")
    }

    override fun Tensor<Ternary, Byte>.flatten(startDim: Int, endDim: Int): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Flatten not yet implemented for Ternary tensors")
    }

    override fun Tensor<Ternary, Byte>.reshape(newShape: Shape): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Reshape not yet implemented for Ternary tensors")
    }

    override fun Tensor<Ternary, Byte>.reshape(vararg dimensions: Int): Tensor<Ternary, Byte> {
        throw UnsupportedOperationException("Reshape not yet implemented for Ternary tensors")
    }

    public companion object {
        /**
         * Creates a tensor from packed byte data.
         */
        public fun fromPackedByteArray(shape: Shape, data: ByteArray): CpuTensorTernary {
            val expectedBytes = (shape.volume + 3) / 4
            require(data.size == expectedBytes) {
                "Data size ${data.size} doesn't match expected packed size $expectedBytes bytes"
            }
            return CpuTensorTernary(shape, data)
        }
    }
}

/**
 * A CPU-based tensor for FP16/Float values with 16-bit floating point storage.
 * Each value uses 2 bytes and is converted to FP32 for operations.
 * This implementation converts FP16 data to FP32 for processing.
 */
public class CpuTensorFP16(
    override val shape: Shape,
    internal val data: FloatArray  // Converted FP32 data
) : TensorFP16 {

    // TensorData implementation
    override val strides: IntArray = shape.computeStrides()
    override val offset: Int = 0
    override val isContiguous: Boolean = true
    
    init {
        require(shape.rank in 1..4) {
            "Only 1-4 dimensional tensors are supported, got ${shape.rank}"
        }
        require(data.size == shape.volume) {
            "Data size ${data.size} doesn't match shape volume ${shape.volume}"
        }
    }

    override fun get(vararg indices: Int): Float {
        val linearIndex = shape.index(indices)
        return data[linearIndex]
    }
    
    override fun copyTo(dest: Array<Float>, destOffset: Int) {
        for (i in data.indices) {
            dest[destOffset + i] = data[i]
        }
    }
    
    override fun slice(ranges: IntArray): TensorData<FP16, Float> {
        return slice_impl(ranges)
    }
    
    private fun slice_impl(ranges: IntArray): CpuTensorFP16 {
        require(ranges.size == shape.rank * 2) {
            "Ranges array must have size ${shape.rank * 2} (start,end pairs), got ${ranges.size}"
        }
        
        // Parse ranges and calculate new shape
        val sliceRanges = mutableListOf<Pair<Int, Int>>()
        for (i in 0 until shape.rank) {
            val start = ranges[i * 2]
            val end = ranges[i * 2 + 1]
            val dimSize = shape.dimensions[i]
            
            require(start >= 0 && start < dimSize) {
                "Start index $start out of bounds for dimension $i (size $dimSize)"
            }
            require(end > start && end <= dimSize) {
                "End index $end must be > start ($start) and <= dimension size ($dimSize)"
            }
            
            sliceRanges.add(start to end)
        }
        
        val newDimensions = sliceRanges.map { (start, end) -> end - start }.toIntArray()
        val newShape = Shape(newDimensions)
        val newData = FloatArray(newShape.volume)
        
        // Extract sliced values
        var destLinearIndex = 0
        
        when (shape.rank) {
            1 -> {
                val (start0, end0) = sliceRanges[0]
                for (i in start0 until end0) {
                    newData[destLinearIndex++] = get(i)
                }
            }
            2 -> {
                val (start0, end0) = sliceRanges[0]
                val (start1, end1) = sliceRanges[1]
                for (i in start0 until end0) {
                    for (j in start1 until end1) {
                        newData[destLinearIndex++] = get(i, j)
                    }
                }
            }
            else -> throw UnsupportedOperationException("Slicing not fully implemented for ${shape.rank}D tensors")
        }
        
        return CpuTensorFP16(newShape, newData)
    }
    
    override fun materialize(): TensorData<FP16, Float> = this
    
    private fun Shape.computeStrides(): IntArray {
        if (dimensions.isEmpty()) return intArrayOf()
        val strides = IntArray(dimensions.size)
        strides[dimensions.size - 1] = 1
        for (i in dimensions.size - 2 downTo 0) {
            strides[i] = strides[i + 1] * dimensions[i + 1]
        }
        return strides
    }

    // Placeholder operations - similar to other tensor types
    override fun matmul(a: Tensor<FP16, Float>, b: Tensor<FP16, Float>): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Matrix multiplication not yet implemented for FP16 tensors")
    }

    override fun matmul4d(a: Tensor<FP16, Float>, b: Tensor<FP16, Float>): Tensor<FP16, Float> {
        throw UnsupportedOperationException("4D matrix multiplication not yet implemented for FP16 tensors")
    }

    override fun scale(a: Tensor<FP16, Float>, scalar: Double): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Scale operation not yet implemented for FP16 tensors")
    }

    override fun dot(a: Tensor<FP16, Float>, b: Tensor<FP16, Float>): Double {
        throw UnsupportedOperationException("Dot product not yet implemented for FP16 tensors")
    }

    override fun Tensor<FP16, Float>.plus(other: Tensor<FP16, Float>): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Addition not yet implemented for FP16 tensors")
    }

    override fun Tensor<FP16, Float>.minus(other: Tensor<FP16, Float>): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Subtraction not yet implemented for FP16 tensors")
    }

    override fun Tensor<FP16, Float>.times(other: Tensor<FP16, Float>): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Multiplication not yet implemented for FP16 tensors")
    }

    override fun Tensor<FP16, Float>.div(other: Tensor<FP16, Float>): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Division not yet implemented for FP16 tensors")
    }

    override fun Tensor<FP16, Float>.plus(scalar: Int): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Scalar addition not yet implemented for FP16 tensors")
    }

    override fun Tensor<FP16, Float>.minus(scalar: Int): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Scalar subtraction not yet implemented for FP16 tensors")
    }

    override fun Tensor<FP16, Float>.times(scalar: Int): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Scalar multiplication not yet implemented for FP16 tensors")
    }

    override fun Tensor<FP16, Float>.div(scalar: Int): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Scalar division not yet implemented for FP16 tensors")
    }

    override fun Tensor<FP16, Float>.plus(scalar: Float): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Float scalar addition not yet implemented for FP16 tensors")
    }

    override fun Tensor<FP16, Float>.minus(scalar: Float): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Float scalar subtraction not yet implemented for FP16 tensors")
    }

    override fun Tensor<FP16, Float>.times(scalar: Float): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Float scalar multiplication not yet implemented for FP16 tensors")
    }

    override fun Tensor<FP16, Float>.div(scalar: Float): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Float scalar division not yet implemented for FP16 tensors")
    }

    override fun Tensor<FP16, Float>.plus(scalar: Double): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Double scalar addition not yet implemented for FP16 tensors")
    }

    override fun Tensor<FP16, Float>.minus(scalar: Double): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Double scalar subtraction not yet implemented for FP16 tensors")
    }

    override fun Tensor<FP16, Float>.times(scalar: Double): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Double scalar multiplication not yet implemented for FP16 tensors")
    }

    override fun Tensor<FP16, Float>.div(scalar: Double): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Double scalar division not yet implemented for FP16 tensors")
    }

    override fun Double.plus(t: Tensor<FP16, Float>): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Scalar-tensor addition not yet implemented for FP16 tensors")
    }

    override fun Double.minus(t: Tensor<FP16, Float>): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Scalar-tensor subtraction not yet implemented for FP16 tensors")
    }

    override fun Double.times(t: Tensor<FP16, Float>): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Scalar-tensor multiplication not yet implemented for FP16 tensors")
    }

    override fun Double.div(t: Tensor<FP16, Float>): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Scalar-tensor division not yet implemented for FP16 tensors")
    }

    override fun Tensor<FP16, Float>.t(): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Transpose not yet implemented for FP16 tensors")
    }

    override fun Tensor<FP16, Float>.relu(): Tensor<FP16, Float> {
        throw UnsupportedOperationException("ReLU not yet implemented for FP16 tensors")
    }

    override fun Tensor<FP16, Float>.sigmoid(): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Sigmoid not yet implemented for FP16 tensors")
    }

    override fun Tensor<FP16, Float>.tanh(): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Tanh not yet implemented for FP16 tensors")
    }

    override fun Tensor<FP16, Float>.softmax(dimension: Int): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Softmax not yet implemented for FP16 tensors")
    }

    override fun Tensor<FP16, Float>.flatten(startDim: Int, endDim: Int): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Flatten not yet implemented for FP16 tensors")
    }

    override fun Tensor<FP16, Float>.reshape(newShape: Shape): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Reshape not yet implemented for FP16 tensors")
    }

    override fun Tensor<FP16, Float>.reshape(vararg dimensions: Int): Tensor<FP16, Float> {
        throw UnsupportedOperationException("Reshape not yet implemented for FP16 tensors")
    }

    public companion object {
        /**
         * Creates a tensor from FP16 byte data by converting to FP32.
         */
        public fun fromFP16ByteArray(shape: Shape, data: ByteArray): CpuTensorFP16 {
            val expectedBytes = shape.volume * 2  // 2 bytes per FP16 value
            require(data.size == expectedBytes) {
                "Data size ${data.size} doesn't match expected FP16 size $expectedBytes bytes"
            }
            
            // Convert FP16 bytes to FP32 floats
            val floatData = FloatArray(shape.volume)
            for (i in 0 until shape.volume) {
                val fp16Bytes = (data[i * 2].toInt() and 0xFF) or ((data[i * 2 + 1].toInt() and 0xFF) shl 8)
                floatData[i] = fp16ToFp32(fp16Bytes)
            }
            
            return CpuTensorFP16(shape, floatData)
        }
        
        /**
         * Converts a 16-bit FP16 value to 32-bit FP32 value.
         * This is a basic implementation - production code might need more sophisticated handling.
         */
        private fun fp16ToFp32(fp16: Int): Float {
            val sign = (fp16 shr 15) and 0x1
            val exponent = (fp16 shr 10) and 0x1F
            val mantissa = fp16 and 0x3FF
            
            return when {
                exponent == 0 -> {
                    // Zero or denormal
                    if (mantissa == 0) {
                        if (sign == 1) -0.0f else 0.0f
                    } else {
                        // Denormal - convert to normal FP32
                        val normalizedMantissa = mantissa.toFloat() / 1024.0f
                        val value = normalizedMantissa * 2.0.pow(-14.0).toFloat()
                        if (sign == 1) -value else value
                    }
                }
                exponent == 31 -> {
                    // Infinity or NaN  
                    if (mantissa == 0) {
                        if (sign == 1) Float.NEGATIVE_INFINITY else Float.POSITIVE_INFINITY
                    } else {
                        Float.NaN
                    }
                }
                else -> {
                    // Normal number
                    val fp32Exponent = exponent - 15 + 127  // Convert bias
                    val fp32Mantissa = mantissa shl 13  // Shift mantissa
                    val fp32Bits = (sign shl 31) or (fp32Exponent shl 23) or fp32Mantissa
                    Float.fromBits(fp32Bits)
                }
            }
        }
    }
}