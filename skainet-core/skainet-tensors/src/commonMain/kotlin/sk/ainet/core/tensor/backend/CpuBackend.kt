package sk.ainet.core.tensor.backend

import sk.ainet.core.tensor.*

/**
 * Object to initialize default tensor factories.
 */
internal object TensorFactoryInitializer {
    init {
        BackendDispatcher.initializeDefaultBackends()
    }
    
    fun ensureInitialized() {
        // Just accessing this object ensures the init block runs
    }
}

/**
 * Convenient type aliases for different tensor types.
 */
public typealias TensorFP32 = Tensor<FP32, Float>
public typealias TensorInt8 = Tensor<Int8, Byte>
public typealias TensorInt32 = Tensor<Int32, Int>

/**
 * CPU-based tensor implementation for FP32/Float values.
 */
public class CpuTensorFP32(
    private val tensorData: TensorData<FP32, Float>
) : TensorFP32 {

    public constructor(shape: Shape, data: FloatArray) : this(
        DenseTensorData<FP32, Float>(shape, data.toTypedArray())
    )

    override val data: TensorData<FP32, Float> = tensorData
    override val ops: TensorOps<FP32, Float, Tensor<FP32, Float>> = CpuBackendFP32()

    init {
        require(data.shape.rank in 1..4) {
            "Only 1-4 dimensional tensors are supported, got ${data.shape.rank}"
        }
    }

    internal val floatArrayData: FloatArray get() {
        return when (tensorData) {
            is DenseTensorData -> {
                val floatArray = FloatArray(data.shape.volume)
                val arrayFloat = Array<Float>(data.shape.volume) { 0f }
                tensorData.copyTo(arrayFloat, 0)
                for (i in arrayFloat.indices) {
                    floatArray[i] = arrayFloat[i]
                }
                floatArray
            }
            else -> {
                val materialized = tensorData.materialize()
                val floatArray = FloatArray(data.shape.volume)
                val arrayFloat = Array<Float>(data.shape.volume) { 0f }
                materialized.copyTo(arrayFloat, 0)
                for (i in arrayFloat.indices) {
                    floatArray[i] = arrayFloat[i]
                }
                floatArray
            }
        }
    }

    override fun toString(): String = "CpuTensorFP32(${shape})"

    public companion object {
        public fun fromArray(shape: Shape, data: FloatArray): CpuTensorFP32 {
            require(data.size == shape.volume) {
                "Data size ${data.size} doesn't match shape volume ${shape.volume}"
            }
            return CpuTensorFP32(shape, data)
        }

        public fun zeros(shape: Shape): CpuTensorFP32 = CpuTensorFP32(shape, FloatArray(shape.volume))
        public fun ones(shape: Shape): CpuTensorFP32 = CpuTensorFP32(shape, FloatArray(shape.volume) { 1.0f })
        public fun full(shape: Shape, value: Float): CpuTensorFP32 = CpuTensorFP32(shape, FloatArray(shape.volume) { value })
        
        public fun fromNestedList(data: List<List<Float>>): CpuTensorFP32 {
            require(data.isNotEmpty()) { "Data cannot be empty" }
            require(data[0].isNotEmpty()) { "Rows cannot be empty" }
            
            val rows = data.size
            val cols = data[0].size
            
            // Validate all rows have same number of columns
            for (i in 1 until rows) {
                require(data[i].size == cols) { 
                    "All rows must have the same number of columns. Row 0 has $cols columns, but row $i has ${data[i].size} columns" 
                }
            }
            
            val flatData = FloatArray(rows * cols)
            for (i in 0 until rows) {
                for (j in 0 until cols) {
                    flatData[i * cols + j] = data[i][j]
                }
            }
            
            return CpuTensorFP32(Shape(rows, cols), flatData)
        }
    }
}

/**
 * CPU-based tensor implementation for Int8/Byte values.
 */
public class CpuTensorInt8(
    private val tensorData: TensorData<Int8, Byte>
) : TensorInt8 {

    public constructor(shape: Shape, data: ByteArray) : this(
        DenseTensorData<Int8, Byte>(shape, data.toTypedArray())
    )

    override val data: TensorData<Int8, Byte> = tensorData
    override val ops: TensorOps<Int8, Byte, Tensor<Int8, Byte>> = CpuBackendInt8()

    internal val byteArrayData: ByteArray get() {
        return when (tensorData) {
            is DenseTensorData -> {
                val byteArray = ByteArray(data.shape.volume)
                val arrayByte = Array<Byte>(data.shape.volume) { 0 }
                tensorData.copyTo(arrayByte, 0)
                for (i in arrayByte.indices) {
                    byteArray[i] = arrayByte[i]
                }
                byteArray
            }
            else -> {
                val materialized = tensorData.materialize()
                val byteArray = ByteArray(data.shape.volume)
                val arrayByte = Array<Byte>(data.shape.volume) { 0 }
                materialized.copyTo(arrayByte, 0)
                for (i in arrayByte.indices) {
                    byteArray[i] = arrayByte[i]
                }
                byteArray
            }
        }
    }

    override fun toString(): String = "CpuTensorInt8(${shape})"

    public companion object {
        public fun fromArray(shape: Shape, data: ByteArray): CpuTensorInt8 {
            require(data.size == shape.volume) {
                "Data size ${data.size} doesn't match shape volume ${shape.volume}"
            }
            return CpuTensorInt8(shape, data)
        }

        public fun zeros(shape: Shape): CpuTensorInt8 = CpuTensorInt8(shape, ByteArray(shape.volume))
    }
}

/**
 * CPU-based tensor implementation for Int32/Int values.
 */
public class CpuTensorInt32(
    private val tensorData: TensorData<Int32, Int>
) : TensorInt32 {

    public constructor(shape: Shape, data: IntArray) : this(
        DenseTensorData<Int32, Int>(shape, data.toTypedArray())
    )

    override val data: TensorData<Int32, Int> = tensorData
    override val ops: TensorOps<Int32, Int, Tensor<Int32, Int>> = CpuBackendInt32()

    internal val intArrayData: IntArray get() {
        return when (tensorData) {
            is DenseTensorData -> {
                val intArray = IntArray(data.shape.volume)
                val arrayInt = Array<Int>(data.shape.volume) { 0 }
                tensorData.copyTo(arrayInt, 0)
                for (i in arrayInt.indices) {
                    intArray[i] = arrayInt[i]
                }
                intArray
            }
            else -> {
                val materialized = tensorData.materialize()
                val intArray = IntArray(data.shape.volume)
                val arrayInt = Array<Int>(data.shape.volume) { 0 }
                materialized.copyTo(arrayInt, 0)
                for (i in arrayInt.indices) {
                    intArray[i] = arrayInt[i]
                }
                intArray
            }
        }
    }

    override fun toString(): String = "CpuTensorInt32(${shape})"

    public companion object {
        public fun fromArray(shape: Shape, data: IntArray): CpuTensorInt32 {
            require(data.size == shape.volume) {
                "Data size ${data.size} doesn't match shape volume ${shape.volume}"
            }
            return CpuTensorInt32(shape, data)
        }

        public fun zeros(shape: Shape): CpuTensorInt32 = CpuTensorInt32(shape, IntArray(shape.volume))
        
        public fun ones(shape: Shape): CpuTensorInt32 = CpuTensorInt32(shape, IntArray(shape.volume) { 1 })
        
        public fun fromNestedList(data: List<List<Int>>): CpuTensorInt32 {
            require(data.isNotEmpty()) { "Data cannot be empty" }
            require(data[0].isNotEmpty()) { "Rows cannot be empty" }
            
            val rows = data.size
            val cols = data[0].size
            
            // Validate all rows have same number of columns
            for (i in 1 until rows) {
                require(data[i].size == cols) { 
                    "All rows must have the same number of columns. Row 0 has $cols columns, but row $i has ${data[i].size} columns" 
                }
            }
            
            val flatData = IntArray(rows * cols)
            for (i in 0 until rows) {
                for (j in 0 until cols) {
                    flatData[i * cols + j] = data[i][j]
                }
            }
            
            return CpuTensorInt32(Shape(rows, cols), flatData)
        }
    }
}

/**
 * CPU backend implementation for FP32 tensors.
 */
public class CpuBackendFP32 : ComputeBackend<FP32, Float, Tensor<FP32, Float>> {
    override val name: String = "CpuBackendFP32"

    override fun matmul(a: Tensor<FP32, Float>, b: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(a is CpuTensorFP32 && b is CpuTensorFP32) { "Both tensors must be CpuTensorFP32" }
        require(a.shape.rank == 2 && b.shape.rank == 2) { "Matrix multiplication requires 2D tensors" }
        require(a.shape[1] == b.shape[0]) { "Matrix dimensions don't match for multiplication" }

        val rows = a.shape[0]
        val cols = b.shape[1]
        val inner = a.shape[1]
        val result = FloatArray(rows * cols)

        val aData = a.floatArrayData
        val bData = b.floatArrayData

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                var sum = 0.0f
                for (k in 0 until inner) {
                    sum += aData[i * inner + k] * bData[k * cols + j]
                }
                result[i * cols + j] = sum
            }
        }

        return CpuTensorFP32(Shape(rows, cols), result)
    }

    override fun dot(a: Tensor<FP32, Float>, b: Tensor<FP32, Float>): Double {
        require(a is CpuTensorFP32 && b is CpuTensorFP32) { "Both tensors must be CpuTensorFP32" }
        require(a.shape == b.shape) { "Tensors must have same shape for dot product" }

        val aData = a.floatArrayData
        val bData = b.floatArrayData
        var sum = 0.0
        for (i in aData.indices) {
            sum += aData[i] * bData[i]
        }
        return sum
    }

    override fun scale(tensor: Tensor<FP32, Float>, scalar: Double): Tensor<FP32, Float> {
        require(tensor is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val data = tensor.floatArrayData
        val result = FloatArray(data.size) { data[it] * scalar.toFloat() }
        return CpuTensorFP32(tensor.shape, result)
    }

    override fun matmul4d(a: Tensor<FP32, Float>, b: Tensor<FP32, Float>): Tensor<FP32, Float> {
        // Basic implementation - delegate to matmul for now
        return matmul(a, b)
    }

    // Tensor-Tensor operations
    override fun Tensor<FP32, Float>.plus(other: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(this is CpuTensorFP32 && other is CpuTensorFP32) { "Both tensors must be CpuTensorFP32" }
        
        // Check if broadcasting is needed
        if (this.shape == other.shape) {
            // Same shape - direct element-wise addition
            val aData = this.floatArrayData
            val bData = other.floatArrayData
            val result = FloatArray(aData.size) { aData[it] + bData[it] }
            return CpuTensorFP32(this.shape, result)
        } else {
            // Broadcasting required
            if (!canBroadcast(this.shape, other.shape)) {
                throw IllegalArgumentException("Cannot broadcast shapes ${this.shape} and ${other.shape}")
            }
            
            val resultShape = getBroadcastShape(this.shape, other.shape)
            val result = FloatArray(resultShape.volume)
            val aData = this.floatArrayData
            val bData = other.floatArrayData
            
            for (i in 0 until resultShape.volume) {
                val aIndex = broadcastIndex(i, resultShape, this.shape)
                val bIndex = broadcastIndex(i, resultShape, other.shape)
                result[i] = aData[aIndex] + bData[bIndex]
            }
            
            return CpuTensorFP32(resultShape, result)
        }
    }
    
    // Utility functions for broadcasting
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
        
        return Shape(resultDims)
    }
    
    private fun broadcastIndex(flatIndex: Int, resultShape: Shape, originalShape: Shape): Int {
        val resultRank = resultShape.rank
        val originalRank = originalShape.rank
        
        var index = 0
        var remaining = flatIndex
        val resultDims = resultShape.dimensions
        val originalDims = originalShape.dimensions
        
        for (i in 0 until resultRank) {
            val resultDim = resultDims[i]
            val originalDim = if (i >= resultRank - originalRank) originalDims[i - (resultRank - originalRank)] else 1
            
            val divisor = resultShape.volume / resultShape.cumulativeProduct(i + 1)
            val coord = remaining / divisor
            remaining %= divisor
            
            val originalCoord = if (originalDim == 1) 0 else coord
            val originalDivisor = originalShape.volume / originalShape.cumulativeProduct(maxOf(0, i - (resultRank - originalRank)) + 1)
            index += originalCoord * originalDivisor
        }
        
        return index
    }
    
    // Helper extension for cumulative product calculation
    private fun Shape.cumulativeProduct(upTo: Int): Int {
        var product = 1
        for (i in upTo until rank) {
            product *= dimensions[i]
        }
        return product
    }

    override fun Tensor<FP32, Float>.minus(other: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(this is CpuTensorFP32 && other is CpuTensorFP32) { "Both tensors must be CpuTensorFP32" }
        
        // Check if broadcasting is needed
        if (this.shape == other.shape) {
            // Same shape - direct element-wise subtraction
            val aData = this.floatArrayData
            val bData = other.floatArrayData
            val result = FloatArray(aData.size) { aData[it] - bData[it] }
            return CpuTensorFP32(this.shape, result)
        } else {
            // Broadcasting required
            if (!canBroadcast(this.shape, other.shape)) {
                throw IllegalArgumentException("Cannot broadcast shapes ${this.shape} and ${other.shape}")
            }
            
            val resultShape = getBroadcastShape(this.shape, other.shape)
            val result = FloatArray(resultShape.volume)
            val aData = this.floatArrayData
            val bData = other.floatArrayData
            
            for (i in 0 until resultShape.volume) {
                val aIndex = broadcastIndex(i, resultShape, this.shape)
                val bIndex = broadcastIndex(i, resultShape, other.shape)
                result[i] = aData[aIndex] - bData[bIndex]
            }
            
            return CpuTensorFP32(resultShape, result)
        }
    }

    override fun Tensor<FP32, Float>.times(other: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(this is CpuTensorFP32 && other is CpuTensorFP32) { "Both tensors must be CpuTensorFP32" }
        
        // Check if broadcasting is needed
        if (this.shape == other.shape) {
            // Same shape - direct element-wise multiplication
            val aData = this.floatArrayData
            val bData = other.floatArrayData
            val result = FloatArray(aData.size) { aData[it] * bData[it] }
            return CpuTensorFP32(this.shape, result)
        } else {
            // Broadcasting required
            if (!canBroadcast(this.shape, other.shape)) {
                throw IllegalArgumentException("Cannot broadcast shapes ${this.shape} and ${other.shape}")
            }
            
            val resultShape = getBroadcastShape(this.shape, other.shape)
            val result = FloatArray(resultShape.volume)
            val aData = this.floatArrayData
            val bData = other.floatArrayData
            
            for (i in 0 until resultShape.volume) {
                val aIndex = broadcastIndex(i, resultShape, this.shape)
                val bIndex = broadcastIndex(i, resultShape, other.shape)
                result[i] = aData[aIndex] * bData[bIndex]
            }
            
            return CpuTensorFP32(resultShape, result)
        }
    }

    override fun Tensor<FP32, Float>.div(other: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(this is CpuTensorFP32 && other is CpuTensorFP32) { "Both tensors must be CpuTensorFP32" }
        
        // Check if broadcasting is needed
        if (this.shape == other.shape) {
            // Same shape - direct element-wise division
            val aData = this.floatArrayData
            val bData = other.floatArrayData
            val result = FloatArray(aData.size) { aData[it] / bData[it] }
            return CpuTensorFP32(this.shape, result)
        } else {
            // Broadcasting required
            if (!canBroadcast(this.shape, other.shape)) {
                throw IllegalArgumentException("Cannot broadcast shapes ${this.shape} and ${other.shape}")
            }
            
            val resultShape = getBroadcastShape(this.shape, other.shape)
            val result = FloatArray(resultShape.volume)
            val aData = this.floatArrayData
            val bData = other.floatArrayData
            
            for (i in 0 until resultShape.volume) {
                val aIndex = broadcastIndex(i, resultShape, this.shape)
                val bIndex = broadcastIndex(i, resultShape, other.shape)
                result[i] = aData[aIndex] / bData[bIndex]
            }
            
            return CpuTensorFP32(resultShape, result)
        }
    }

    // Tensor-Scalar operations
    override fun Tensor<FP32, Float>.plus(scalar: Int): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val data = this.floatArrayData
        val result = FloatArray(data.size) { data[it] + scalar }
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.minus(scalar: Int): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val data = this.floatArrayData
        val result = FloatArray(data.size) { data[it] - scalar }
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.times(scalar: Int): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val data = this.floatArrayData
        val result = FloatArray(data.size) { data[it] * scalar }
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.div(scalar: Int): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val data = this.floatArrayData
        val result = FloatArray(data.size) { data[it] / scalar }
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.plus(scalar: Float): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val data = this.floatArrayData
        val result = FloatArray(data.size) { data[it] + scalar }
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.minus(scalar: Float): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val data = this.floatArrayData
        val result = FloatArray(data.size) { data[it] - scalar }
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.times(scalar: Float): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val data = this.floatArrayData
        val result = FloatArray(data.size) { data[it] * scalar }
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.div(scalar: Float): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val data = this.floatArrayData
        val result = FloatArray(data.size) { data[it] / scalar }
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.plus(scalar: Double): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val data = this.floatArrayData
        val result = FloatArray(data.size) { data[it] + scalar.toFloat() }
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.minus(scalar: Double): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val data = this.floatArrayData
        val result = FloatArray(data.size) { data[it] - scalar.toFloat() }
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.times(scalar: Double): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val data = this.floatArrayData
        val result = FloatArray(data.size) { data[it] * scalar.toFloat() }
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.div(scalar: Double): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val data = this.floatArrayData
        val result = FloatArray(data.size) { data[it] / scalar.toFloat() }
        return CpuTensorFP32(this.shape, result)
    }

    // Scalar-Tensor operations
    override fun Double.plus(t: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(t is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val data = t.floatArrayData
        val result = FloatArray(data.size) { this.toFloat() + data[it] }
        return CpuTensorFP32(t.shape, result)
    }

    override fun Double.minus(t: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(t is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val data = t.floatArrayData
        val result = FloatArray(data.size) { this.toFloat() - data[it] }
        return CpuTensorFP32(t.shape, result)
    }

    override fun Double.times(t: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(t is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val data = t.floatArrayData
        val result = FloatArray(data.size) { this.toFloat() * data[it] }
        return CpuTensorFP32(t.shape, result)
    }

    override fun Double.div(t: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(t is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val data = t.floatArrayData
        val result = FloatArray(data.size) { this.toFloat() / data[it] }
        return CpuTensorFP32(t.shape, result)
    }

    // Mathematical functions
    override fun Tensor<FP32, Float>.t(): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        require(this.shape.rank == 2) { "Transpose only supported for 2D tensors" }
        val rows = this.shape[0]
        val cols = this.shape[1]
        val data = this.floatArrayData
        val result = FloatArray(data.size)
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result[j * rows + i] = data[i * cols + j]
            }
        }
        return CpuTensorFP32(Shape(cols, rows), result)
    }

    override fun Tensor<FP32, Float>.relu(): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val data = this.floatArrayData
        val result = FloatArray(data.size) { maxOf(0.0f, data[it]) }
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.sigmoid(): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val data = this.floatArrayData
        val result = FloatArray(data.size) { 1.0f / (1.0f + kotlin.math.exp(-data[it])) }
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.tanh(): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val data = this.floatArrayData
        val result = FloatArray(data.size) { kotlin.math.tanh(data[it]) }
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.softmax(dimension: Int): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        // Basic implementation for 1D case
        val data = this.floatArrayData
        val maxVal = data.maxOrNull() ?: 0.0f
        val expValues = FloatArray(data.size) { kotlin.math.exp(data[it] - maxVal) }
        val sum = expValues.sum()
        val result = FloatArray(data.size) { expValues[it] / sum }
        return CpuTensorFP32(this.shape, result)
    }

    override fun Tensor<FP32, Float>.flatten(startDim: Int, endDim: Int): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        return CpuTensorFP32(Shape(this.shape.volume), this.floatArrayData.copyOf())
    }

    override fun Tensor<FP32, Float>.reshape(newShape: Shape): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        require(newShape.volume == this.shape.volume) { "New shape volume must match original" }
        return CpuTensorFP32(newShape, this.floatArrayData.copyOf())
    }

    override fun Tensor<FP32, Float>.reshape(vararg dimensions: Int): Tensor<FP32, Float> {
        val newShape = Shape(*dimensions)
        return this.reshape(newShape)
    }

}

/**
 * CPU backend implementation for Int8 tensors.
 */
public class CpuBackendInt8 : ComputeBackend<Int8, Byte, Tensor<Int8, Byte>> {
    override val name: String = "CpuBackendInt8"

    private fun clampToByte(value: Double): Byte = value.coerceIn(-128.0, 127.0).toInt().toByte()

    override fun matmul(a: Tensor<Int8, Byte>, b: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        require(a is CpuTensorInt8 && b is CpuTensorInt8) { "Both tensors must be CpuTensorInt8" }
        require(a.shape.rank == 2 && b.shape.rank == 2) { "Matrix multiplication requires 2D tensors" }
        require(a.shape[1] == b.shape[0]) { "Matrix dimensions don't match for multiplication" }

        val rows = a.shape[0]
        val cols = b.shape[1]
        val inner = a.shape[1]
        val result = ByteArray(rows * cols)

        val aData = a.byteArrayData
        val bData = b.byteArrayData

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                var sum = 0
                for (k in 0 until inner) {
                    sum += aData[i * inner + k] * bData[k * cols + j]
                }
                result[i * cols + j] = clampToByte(sum.toDouble())
            }
        }

        return CpuTensorInt8(Shape(rows, cols), result)
    }

    override fun dot(a: Tensor<Int8, Byte>, b: Tensor<Int8, Byte>): Double {
        require(a is CpuTensorInt8 && b is CpuTensorInt8) { "Both tensors must be CpuTensorInt8" }
        require(a.shape == b.shape) { "Tensors must have same shape for dot product" }

        val aData = a.byteArrayData
        val bData = b.byteArrayData
        var sum = 0.0
        for (i in aData.indices) {
            sum += aData[i] * bData[i]
        }
        return sum
    }

    override fun scale(tensor: Tensor<Int8, Byte>, scalar: Double): Tensor<Int8, Byte> {
        require(tensor is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val data = tensor.byteArrayData
        val result = ByteArray(data.size) { clampToByte(data[it] * scalar) }
        return CpuTensorInt8(tensor.shape, result)
    }

    override fun matmul4d(a: Tensor<Int8, Byte>, b: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        // Basic implementation - delegate to matmul for now
        return matmul(a, b)
    }

    // Tensor-Tensor operations
    override fun Tensor<Int8, Byte>.plus(other: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8 && other is CpuTensorInt8) { "Both tensors must be CpuTensorInt8" }
        require(this.shape == other.shape) { "Tensors must have same shape for addition" }
        val aData = this.byteArrayData
        val bData = other.byteArrayData
        val result = ByteArray(aData.size) { clampToByte((aData[it] + bData[it]).toDouble()) }
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.minus(other: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8 && other is CpuTensorInt8) { "Both tensors must be CpuTensorInt8" }
        require(this.shape == other.shape) { "Tensors must have same shape for subtraction" }
        val aData = this.byteArrayData
        val bData = other.byteArrayData
        val result = ByteArray(aData.size) { clampToByte((aData[it] - bData[it]).toDouble()) }
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.times(other: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8 && other is CpuTensorInt8) { "Both tensors must be CpuTensorInt8" }
        require(this.shape == other.shape) { "Tensors must have same shape for multiplication" }
        val aData = this.byteArrayData
        val bData = other.byteArrayData
        val result = ByteArray(aData.size) { clampToByte((aData[it] * bData[it]).toDouble()) }
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.div(other: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8 && other is CpuTensorInt8) { "Both tensors must be CpuTensorInt8" }
        require(this.shape == other.shape) { "Tensors must have same shape for division" }
        val aData = this.byteArrayData
        val bData = other.byteArrayData
        val result = ByteArray(aData.size) { clampToByte((aData[it] / bData[it]).toDouble()) }
        return CpuTensorInt8(this.shape, result)
    }

    // Tensor-Scalar operations
    override fun Tensor<Int8, Byte>.plus(scalar: Int): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val data = this.byteArrayData
        val result = ByteArray(data.size) { clampToByte((data[it] + scalar).toDouble()) }
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.minus(scalar: Int): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val data = this.byteArrayData
        val result = ByteArray(data.size) { clampToByte((data[it] - scalar).toDouble()) }
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.times(scalar: Int): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val data = this.byteArrayData
        val result = ByteArray(data.size) { clampToByte((data[it] * scalar).toDouble()) }
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.div(scalar: Int): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val data = this.byteArrayData
        val result = ByteArray(data.size) { clampToByte((data[it] / scalar).toDouble()) }
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.plus(scalar: Float): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val data = this.byteArrayData
        val result = ByteArray(data.size) { clampToByte((data[it] + scalar).toDouble()) }
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.minus(scalar: Float): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val data = this.byteArrayData
        val result = ByteArray(data.size) { clampToByte((data[it] - scalar).toDouble()) }
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.times(scalar: Float): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val data = this.byteArrayData
        val result = ByteArray(data.size) { clampToByte((data[it] * scalar).toDouble()) }
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.div(scalar: Float): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val data = this.byteArrayData
        val result = ByteArray(data.size) { clampToByte((data[it] / scalar).toDouble()) }
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.plus(scalar: Double): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val data = this.byteArrayData
        val result = ByteArray(data.size) { clampToByte(data[it] + scalar) }
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.minus(scalar: Double): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val data = this.byteArrayData
        val result = ByteArray(data.size) { clampToByte(data[it] - scalar) }
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.times(scalar: Double): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val data = this.byteArrayData
        val result = ByteArray(data.size) { clampToByte(data[it] * scalar) }
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.div(scalar: Double): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val data = this.byteArrayData
        val result = ByteArray(data.size) { clampToByte(data[it] / scalar) }
        return CpuTensorInt8(this.shape, result)
    }

    // Scalar-Tensor operations
    override fun Double.plus(t: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        require(t is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val data = t.byteArrayData
        val result = ByteArray(data.size) { clampToByte(this + data[it]) }
        return CpuTensorInt8(t.shape, result)
    }

    override fun Double.minus(t: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        require(t is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val data = t.byteArrayData
        val result = ByteArray(data.size) { clampToByte(this - data[it]) }
        return CpuTensorInt8(t.shape, result)
    }

    override fun Double.times(t: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        require(t is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val data = t.byteArrayData
        val result = ByteArray(data.size) { clampToByte(this * data[it]) }
        return CpuTensorInt8(t.shape, result)
    }

    override fun Double.div(t: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        require(t is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val data = t.byteArrayData
        val result = ByteArray(data.size) { clampToByte(this / data[it]) }
        return CpuTensorInt8(t.shape, result)
    }

    // Mathematical functions
    override fun Tensor<Int8, Byte>.t(): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        require(this.shape.rank == 2) { "Transpose only supported for 2D tensors" }
        val rows = this.shape[0]
        val cols = this.shape[1]
        val data = this.byteArrayData
        val result = ByteArray(data.size)
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result[j * rows + i] = data[i * cols + j]
            }
        }
        return CpuTensorInt8(Shape(cols, rows), result)
    }

    override fun Tensor<Int8, Byte>.relu(): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val data = this.byteArrayData
        val result = ByteArray(data.size) { maxOf(0.toByte(), data[it]) }
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.sigmoid(): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val data = this.byteArrayData
        val result = ByteArray(data.size) { 
            clampToByte(127.0 / (1.0 + kotlin.math.exp(-data[it].toDouble() / 127.0)))
        }
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.tanh(): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        val data = this.byteArrayData
        val result = ByteArray(data.size) { 
            clampToByte(127.0 * kotlin.math.tanh(data[it].toDouble() / 127.0))
        }
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.softmax(dimension: Int): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        // Basic implementation for 1D case
        val data = this.byteArrayData
        val maxVal = data.maxOrNull()?.toDouble() ?: 0.0
        val expValues = DoubleArray(data.size) { kotlin.math.exp((data[it] - maxVal) / 127.0) }
        val sum = expValues.sum()
        val result = ByteArray(data.size) { clampToByte(127.0 * expValues[it] / sum) }
        return CpuTensorInt8(this.shape, result)
    }

    override fun Tensor<Int8, Byte>.flatten(startDim: Int, endDim: Int): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        return CpuTensorInt8(Shape(this.shape.volume), this.byteArrayData.copyOf())
    }

    override fun Tensor<Int8, Byte>.reshape(newShape: Shape): Tensor<Int8, Byte> {
        require(this is CpuTensorInt8) { "Tensor must be CpuTensorInt8" }
        require(newShape.volume == this.shape.volume) { "New shape volume must match original" }
        return CpuTensorInt8(newShape, this.byteArrayData.copyOf())
    }

    override fun Tensor<Int8, Byte>.reshape(vararg dimensions: Int): Tensor<Int8, Byte> {
        val newShape = Shape(*dimensions)
        return this.reshape(newShape)
    }

}

/**
 * CPU backend implementation for Int32 tensors.
 */
public class CpuBackendInt32 : ComputeBackend<Int32, Int, Tensor<Int32, Int>> {
    override val name: String = "CpuBackendInt32"

    override fun matmul(a: Tensor<Int32, Int>, b: Tensor<Int32, Int>): Tensor<Int32, Int> {
        require(a is CpuTensorInt32 && b is CpuTensorInt32) { "Both tensors must be CpuTensorInt32" }
        require(a.shape.rank == 2 && b.shape.rank == 2) { "Matrix multiplication requires 2D tensors" }
        require(a.shape[1] == b.shape[0]) { "Matrix dimensions don't match for multiplication" }

        val rows = a.shape[0]
        val cols = b.shape[1]
        val inner = a.shape[1]
        val result = IntArray(rows * cols)

        val aData = a.intArrayData
        val bData = b.intArrayData

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                var sum = 0
                for (k in 0 until inner) {
                    sum += aData[i * inner + k] * bData[k * cols + j]
                }
                result[i * cols + j] = sum
            }
        }

        return CpuTensorInt32(Shape(rows, cols), result)
    }

    override fun dot(a: Tensor<Int32, Int>, b: Tensor<Int32, Int>): Double {
        require(a is CpuTensorInt32 && b is CpuTensorInt32) { "Both tensors must be CpuTensorInt32" }
        require(a.shape == b.shape) { "Tensors must have same shape for dot product" }

        val aData = a.intArrayData
        val bData = b.intArrayData
        var sum = 0.0
        for (i in aData.indices) {
            sum += aData[i] * bData[i]
        }
        return sum
    }

    override fun scale(tensor: Tensor<Int32, Int>, scalar: Double): Tensor<Int32, Int> {
        require(tensor is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val data = tensor.intArrayData
        val result = IntArray(data.size) { (data[it] * scalar).toInt() }
        return CpuTensorInt32(tensor.shape, result)
    }

    override fun matmul4d(a: Tensor<Int32, Int>, b: Tensor<Int32, Int>): Tensor<Int32, Int> {
        // Basic implementation - delegate to matmul for now
        return matmul(a, b)
    }

    // Tensor-Tensor operations
    override fun Tensor<Int32, Int>.plus(other: Tensor<Int32, Int>): Tensor<Int32, Int> {
        require(this is CpuTensorInt32 && other is CpuTensorInt32) { "Both tensors must be CpuTensorInt32" }
        require(this.shape == other.shape) { "Tensors must have same shape for addition" }
        val aData = this.intArrayData
        val bData = other.intArrayData
        val result = IntArray(aData.size) { aData[it] + bData[it] }
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.minus(other: Tensor<Int32, Int>): Tensor<Int32, Int> {
        require(this is CpuTensorInt32 && other is CpuTensorInt32) { "Both tensors must be CpuTensorInt32" }
        require(this.shape == other.shape) { "Tensors must have same shape for subtraction" }
        val aData = this.intArrayData
        val bData = other.intArrayData
        val result = IntArray(aData.size) { aData[it] - bData[it] }
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.times(other: Tensor<Int32, Int>): Tensor<Int32, Int> {
        require(this is CpuTensorInt32 && other is CpuTensorInt32) { "Both tensors must be CpuTensorInt32" }
        require(this.shape == other.shape) { "Tensors must have same shape for multiplication" }
        val aData = this.intArrayData
        val bData = other.intArrayData
        val result = IntArray(aData.size) { aData[it] * bData[it] }
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.div(other: Tensor<Int32, Int>): Tensor<Int32, Int> {
        require(this is CpuTensorInt32 && other is CpuTensorInt32) { "Both tensors must be CpuTensorInt32" }
        require(this.shape == other.shape) { "Tensors must have same shape for division" }
        val aData = this.intArrayData
        val bData = other.intArrayData
        val result = IntArray(aData.size) { aData[it] / bData[it] }
        return CpuTensorInt32(this.shape, result)
    }

    // Tensor-Scalar operations
    override fun Tensor<Int32, Int>.plus(scalar: Int): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val data = this.intArrayData
        val result = IntArray(data.size) { data[it] + scalar }
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.minus(scalar: Int): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val data = this.intArrayData
        val result = IntArray(data.size) { data[it] - scalar }
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.times(scalar: Int): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val data = this.intArrayData
        val result = IntArray(data.size) { data[it] * scalar }
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.div(scalar: Int): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val data = this.intArrayData
        val result = IntArray(data.size) { data[it] / scalar }
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.plus(scalar: Float): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val data = this.intArrayData
        val result = IntArray(data.size) { (data[it] + scalar).toInt() }
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.minus(scalar: Float): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val data = this.intArrayData
        val result = IntArray(data.size) { (data[it] - scalar).toInt() }
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.times(scalar: Float): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val data = this.intArrayData
        val result = IntArray(data.size) { (data[it] * scalar).toInt() }
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.div(scalar: Float): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val data = this.intArrayData
        val result = IntArray(data.size) { (data[it] / scalar).toInt() }
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.plus(scalar: Double): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val data = this.intArrayData
        val result = IntArray(data.size) { (data[it] + scalar).toInt() }
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.minus(scalar: Double): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val data = this.intArrayData
        val result = IntArray(data.size) { (data[it] - scalar).toInt() }
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.times(scalar: Double): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val data = this.intArrayData
        val result = IntArray(data.size) { (data[it] * scalar).toInt() }
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.div(scalar: Double): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val data = this.intArrayData
        val result = IntArray(data.size) { (data[it] / scalar).toInt() }
        return CpuTensorInt32(this.shape, result)
    }

    // Scalar-Tensor operations
    override fun Double.plus(t: Tensor<Int32, Int>): Tensor<Int32, Int> {
        require(t is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val data = t.intArrayData
        val result = IntArray(data.size) { (this + data[it]).toInt() }
        return CpuTensorInt32(t.shape, result)
    }

    override fun Double.minus(t: Tensor<Int32, Int>): Tensor<Int32, Int> {
        require(t is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val data = t.intArrayData
        val result = IntArray(data.size) { (this - data[it]).toInt() }
        return CpuTensorInt32(t.shape, result)
    }

    override fun Double.times(t: Tensor<Int32, Int>): Tensor<Int32, Int> {
        require(t is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val data = t.intArrayData
        val result = IntArray(data.size) { (this * data[it]).toInt() }
        return CpuTensorInt32(t.shape, result)
    }

    override fun Double.div(t: Tensor<Int32, Int>): Tensor<Int32, Int> {
        require(t is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val data = t.intArrayData
        val result = IntArray(data.size) { (this / data[it]).toInt() }
        return CpuTensorInt32(t.shape, result)
    }

    // Mathematical functions
    override fun Tensor<Int32, Int>.t(): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        require(this.shape.rank == 2) { "Transpose only supported for 2D tensors" }
        val rows = this.shape[0]
        val cols = this.shape[1]
        val data = this.intArrayData
        val result = IntArray(data.size)
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result[j * rows + i] = data[i * cols + j]
            }
        }
        return CpuTensorInt32(Shape(cols, rows), result)
    }

    override fun Tensor<Int32, Int>.relu(): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val data = this.intArrayData
        val result = IntArray(data.size) { maxOf(0, data[it]) }
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.sigmoid(): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val data = this.intArrayData
        val result = IntArray(data.size) { 
            (Int.MAX_VALUE / (1.0 + kotlin.math.exp(-data[it].toDouble() / Int.MAX_VALUE.toDouble()))).toInt()
        }
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.tanh(): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        val data = this.intArrayData
        val result = IntArray(data.size) { 
            (Int.MAX_VALUE * kotlin.math.tanh(data[it].toDouble() / Int.MAX_VALUE.toDouble())).toInt()
        }
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.softmax(dimension: Int): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        // Basic implementation for 1D case
        val data = this.intArrayData
        val maxVal = data.maxOrNull()?.toDouble() ?: 0.0
        val expValues = DoubleArray(data.size) { kotlin.math.exp((data[it] - maxVal) / Int.MAX_VALUE.toDouble()) }
        val sum = expValues.sum()
        val result = IntArray(data.size) { (Int.MAX_VALUE * expValues[it] / sum).toInt() }
        return CpuTensorInt32(this.shape, result)
    }

    override fun Tensor<Int32, Int>.flatten(startDim: Int, endDim: Int): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        return CpuTensorInt32(Shape(this.shape.volume), this.intArrayData.copyOf())
    }

    override fun Tensor<Int32, Int>.reshape(newShape: Shape): Tensor<Int32, Int> {
        require(this is CpuTensorInt32) { "Tensor must be CpuTensorInt32" }
        require(newShape.volume == this.shape.volume) { "New shape volume must match original" }
        return CpuTensorInt32(newShape, this.intArrayData.copyOf())
    }

    override fun Tensor<Int32, Int>.reshape(vararg dimensions: Int): Tensor<Int32, Int> {
        val newShape = Shape(*dimensions)
        return this.reshape(newShape)
    }

}

/**
 * Convenience function to create a default CPU backend for FP32 tensors.
 */
public fun CpuBackend(): CpuBackendFP32 = CpuBackendFP32()
