package sk.ainet.core.tensor.backend

import sk.ainet.core.tensor.*

/**
 * Mock tensor implementations that support real data creation but use mock operations.
 * These tensors enable loading data from files/arrays while keeping operations unimplemented,
 * allowing for separation between data handling and computation logic.
 */

/**
 * Mock tensor implementation for FP32/Float values that uses MockBackendFP32 for operations.
 */
public class MockTensorFP32(
    private val tensorData: TensorData<FP32, Float>
) : TensorFP32 {

    public constructor(shape: Shape, data: FloatArray) : this(
        DenseTensorData<FP32, Float>(shape, data.toTypedArray())
    )

    override val data: TensorData<FP32, Float> = tensorData
    override val ops: TensorOps<FP32, Float, Tensor<FP32, Float>> = MockBackendFP32()

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

    override fun toString(): String = "MockTensorFP32(${shape})"

    public companion object {
        public fun fromArray(shape: Shape, data: FloatArray): MockTensorFP32 {
            require(data.size == shape.volume) {
                "Data size ${data.size} doesn't match shape volume ${shape.volume}"
            }
            return MockTensorFP32(shape, data)
        }

        public fun zeros(shape: Shape): MockTensorFP32 = MockTensorFP32(shape, FloatArray(shape.volume))
        public fun ones(shape: Shape): MockTensorFP32 = MockTensorFP32(shape, FloatArray(shape.volume) { 1.0f })
        public fun full(shape: Shape, value: Float): MockTensorFP32 = MockTensorFP32(shape, FloatArray(shape.volume) { value })
        
        public fun fromNestedList(data: List<List<Float>>): MockTensorFP32 {
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
            
            return MockTensorFP32(Shape(rows, cols), flatData)
        }
    }
}

/**
 * Mock tensor implementation for Int32/Int values that uses MockBackendInt32 for operations.
 */
public class MockTensorInt32(
    private val tensorData: TensorData<Int32, Int>
) : TensorInt32 {

    public constructor(shape: Shape, data: IntArray) : this(
        DenseTensorData<Int32, Int>(shape, data.toTypedArray())
    )

    override val data: TensorData<Int32, Int> = tensorData
    override val ops: TensorOps<Int32, Int, Tensor<Int32, Int>> = MockBackendInt32()

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

    override fun toString(): String = "MockTensorInt32(${shape})"

    public companion object {
        public fun fromArray(shape: Shape, data: IntArray): MockTensorInt32 {
            require(data.size == shape.volume) {
                "Data size ${data.size} doesn't match shape volume ${shape.volume}"
            }
            return MockTensorInt32(shape, data)
        }

        public fun zeros(shape: Shape): MockTensorInt32 = MockTensorInt32(shape, IntArray(shape.volume))
        public fun ones(shape: Shape): MockTensorInt32 = MockTensorInt32(shape, IntArray(shape.volume) { 1 })
        
        public fun fromNestedList(data: List<List<Int>>): MockTensorInt32 {
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
            
            return MockTensorInt32(Shape(rows, cols), flatData)
        }
    }
}

/**
 * Mock tensor implementation for Int8/Byte values that uses MockBackendInt8 for operations.
 */
public class MockTensorInt8(
    private val tensorData: TensorData<Int8, Byte>
) : TensorInt8 {

    public constructor(shape: Shape, data: ByteArray) : this(
        DenseTensorData<Int8, Byte>(shape, data.toTypedArray())
    )

    override val data: TensorData<Int8, Byte> = tensorData
    override val ops: TensorOps<Int8, Byte, Tensor<Int8, Byte>> = MockBackendInt8()

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

    override fun toString(): String = "MockTensorInt8(${shape})"

    public companion object {
        public fun fromArray(shape: Shape, data: ByteArray): MockTensorInt8 {
            require(data.size == shape.volume) {
                "Data size ${data.size} doesn't match shape volume ${shape.volume}"
            }
            return MockTensorInt8(shape, data)
        }

        public fun zeros(shape: Shape): MockTensorInt8 = MockTensorInt8(shape, ByteArray(shape.volume))
    }
}