package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.memory.DenseFloatTensorArray
import sk.ainet.lang.tensor.memory.DenseByteTensorArray
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import sk.ainet.lang.types.Int4
import sk.ainet.lang.types.Int8
import sk.ainet.lang.types.Ternary
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Low level DenseTensorDataFactory factories.
 */

public class DenseTensorDataFactory {

    private fun calcFlatIndex(shape: Shape, strides: IntArray, vararg indices: Int): Int {
        require(indices.size == shape.dimensions.size) {
            "Number of indices (${indices.size}) must match tensor dimensions (${shape.dimensions.size})"
        }

        var flatIndex = 0
        for (i in indices.indices) {
            require(indices[i] >= 0 && indices[i] < shape.dimensions[i]) {
                "Index ${indices[i]} out of bounds for dimension $i with size ${shape.dimensions[i]}"
            }
            flatIndex += indices[i] * strides[i]
        }
        return flatIndex
    }


    @Suppress("UNCHECKED_CAST")
    public fun <T : DType> fromFloatArray(
        shape: Shape,
        data: FloatArray,
        dtype: T
    ): TensorData<T, Float> {
        return when (dtype) {
            is FP32, FP16 -> {
                class FloatTensorDataImpl(
                    private val denseArray: DenseFloatTensorArray
                ) : TensorData<T, Float>, ItemsAccessor<Float> by denseArray {
                    override val shape: Shape = denseArray.shape
                }
                val denseArray = DenseFloatTensorArray(shape, data.copyOf())
                FloatTensorDataImpl(denseArray) as TensorData<T, Float>
            }

            else -> throw IllegalArgumentException("Unsupported dtype: ${dtype.name}")
        }
    }

    /**
     * Creates a tensor filled with a constant value for the specified dtype.
     *
     * @param T the data type constraint extending DType
     * @param shape the shape of the tensor to create
     * @param value the constant value to fill the tensor with
     * @param dtype the data type instance
     * @return TensorData filled with the constant value
     */
    @Suppress("UNCHECKED_CAST")
    public fun <T : DType> full(shape: Shape, value: Number, dtype: T): TensorData<T, *> {
        return when (dtype) {
            is Int32 -> {
                val intValue = value.toInt()
                val data = IntArray(shape.volume) { intValue }
                createIntTensorData(shape, data) as TensorData<T, *>
            }

            is FP32, FP16 -> {
                val floatValue = value.toFloat()
                val data = FloatArray(shape.volume) { floatValue }
                createFloatTensorData(shape, data, dtype) as TensorData<T, *>
            }

            else -> throw IllegalArgumentException("Unsupported dtype for full: ${dtype.name}")
        }
    }

    /**
     * Creates a tensor filled with random values from uniform distribution [0, 1) for floating point types
     * or [0, maxValue) for integer types.
     *
     * @param T the data type constraint extending DType
     * @param shape the shape of the tensor to create
     * @param dtype the data type instance
     * @param random the random number generator (optional, uses default if not provided)
     * @return TensorData filled with random values
     */
    @Suppress("UNCHECKED_CAST")
    public fun <T : DType, V> randomInt(
        shape: Shape,
        dtype: T,
        random: Random,
        generator: (size: Int, random: Random) -> V
    ): TensorData<T, *> {
        return when (dtype) {
            is Int32 -> {
                val generatedData = generator(shape.volume, random)
                createIntTensorData(shape, generatedData as IntArray) as TensorData<T, *>
            }
            else -> throw IllegalArgumentException("Unsupported dtype for random: ${dtype.name}")
        }
    }

    /**
     * Creates a tensor filled with random values from normal distribution.
     *
     * @param T the data type constraint extending DType
     * @param shape the shape of the tensor to create
     * @param mean the mean of the normal distribution
     * @param std the standard deviation of the normal distribution
     * @param dtype the data type instance
     * @param random the random number generator (optional, uses default if not provided)
     * @return TensorData filled with normal distributed random values
     */
    @Suppress("UNCHECKED_CAST")
    public fun <T : DType> randn(
        shape: Shape,
        mean: Float = 0.0f,
        std: Float = 1.0f,
        dtype: T,
        random: Random = Random.Default
    ): TensorData<T, *> {
        return when (dtype) {
            is FP32, FP16 -> {
                val data = FloatArray(shape.volume) {
                    // Box-Muller transform for normal distribution
                    val u1 = random.nextFloat()
                    val u2 = random.nextFloat()
                    val z0 = sqrt(-2.0 * ln(u1.toDouble())).toFloat() *
                            cos(2.0 * PI * u2.toDouble()).toFloat()
                    mean + std * z0
                }
                createFloatTensorData(shape, data, dtype) as TensorData<T, *>
            }

            else -> throw IllegalArgumentException("Normal distribution only supported for floating point types: ${dtype.name}")
        }
    }

    // Helper methods to create tensor data instances

    private fun createIntTensorData(shape: Shape, data: IntArray): TensorData<Int32, Int> {
        class IntTensorData : TensorData<Int32, Int> {
            private val _data = data.copyOf()
            val strides: IntArray = shape.computeStrides()

            override val shape: Shape
                get() = Shape(shape.dimensions)

            override fun get(vararg indices: Int): Int = _data[calcFlatIndex(shape, strides, *indices)]

            override fun set(vararg indices: Int, value: Int) {
                _data[calcFlatIndex(shape, strides, *indices)] = value
            }
        }
        return IntTensorData()
    }

    @Suppress("UNCHECKED_CAST")
    private fun <T : DType> createFloatTensorData(shape: Shape, data: FloatArray, dtype: T): TensorData<T, Float> {
        class FloatTensorDataImpl(
            private val denseArray: DenseFloatTensorArray
        ) : TensorData<T, Float>, ItemsAccessor<Float> by denseArray {
            override val shape: Shape = denseArray.shape
        }
        val denseArray = DenseFloatTensorArray(shape, data.copyOf())
        return FloatTensorDataImpl(denseArray) as TensorData<T, Float>
    }

    @Suppress("UNCHECKED_CAST")
    private fun <T : DType> createByteTensorData(shape: Shape, data: ByteArray): TensorData<T, Byte> {
        class ByteTensorDataImpl(
            private val denseArray: DenseByteTensorArray
        ) : TensorData<T, Byte>, ItemsAccessor<Byte> by denseArray {
            override val shape: Shape = denseArray.shape
        }
        val denseArray = DenseByteTensorArray(shape, data.copyOf())
        return ByteTensorDataImpl(denseArray) as TensorData<T, Byte>
    }

    // Convenience methods for scalar, vector, and matrix creation

    /**
     * Creates a scalar tensor from a single value.
     */
    @Suppress("UNCHECKED_CAST")
    public fun <T : DType, V> scalar(value: V): TensorData<T, V> {
        return when (value) {
            is Int -> {
                val data = intArrayOf(value)
                createIntTensorData(Shape(1), data) as TensorData<T, V>
            }

            is Float -> {
                val data = floatArrayOf(value)
                createFloatTensorData(Shape(1), data, FP32 as T) as TensorData<T, V>
            }

            is Byte -> {
                val data = intArrayOf(value.toInt())
                createIntTensorData(Shape(1), data) as TensorData<T, V>
            }

            else -> throw IllegalArgumentException("Unsupported value type: ${value!!::class}")
        }
    }

    /**
     * Creates a vector tensor from an array of values.
     */
    @Suppress("UNCHECKED_CAST")
    public fun <T : DType, V> vector(values: Array<V>): TensorData<T, V> {
        return when (values.firstOrNull()) {
            is Int -> {
                val data = values.map { (it as Int) }.toIntArray()
                createIntTensorData(Shape(values.size), data) as TensorData<T, V>
            }

            is Float -> {
                val data = values.map { (it as Float) }.toFloatArray()
                createFloatTensorData(Shape(values.size), data, FP32 as T) as TensorData<T, V>
            }

            is Byte -> {
                val data = values.map { (it as Byte).toInt() }.toIntArray()
                createIntTensorData(Shape(values.size), data) as TensorData<T, V>
            }

            else -> throw IllegalArgumentException("Unsupported value type: ${values.firstOrNull()?.let { it::class }}")
        }
    }

    /**
     * Creates a matrix tensor from multiple rows of values.
     */
    @Suppress("UNCHECKED_CAST")
    public fun <T : DType, V> matrix(vararg rows: Array<V>): TensorData<T, V> {
        require(rows.isNotEmpty()) { "Matrix must have at least one row" }
        val numRows = rows.size
        val numCols = rows[0].size
        require(rows.all { it.size == numCols }) { "All rows must have the same number of columns" }

        return when (rows[0].firstOrNull()) {
            is Int -> {
                val data = rows.flatMap { row -> row.map { (it as Int) } }.toIntArray()
                createIntTensorData(Shape(numRows, numCols), data) as TensorData<T, V>
            }

            is Float -> {
                val data = rows.flatMap { row -> row.map { (it as Float) } }.toFloatArray()
                createFloatTensorData(Shape(numRows, numCols), data, FP32 as T) as TensorData<T, V>
            }

            is Byte -> {
                val data = rows.flatMap { row -> row.map { (it as Byte).toInt() } }.toIntArray()
                createIntTensorData(Shape(numRows, numCols), data) as TensorData<T, V>
            }

            else -> throw IllegalArgumentException(
                "Unsupported value type: ${
                    rows[0].firstOrNull()?.let { it::class }
                }"
            )
        }
    }

    /**
     * Creates a tensor from byte array data for various data types.
     */
    @Suppress("UNCHECKED_CAST")
    public fun <T : DType, V> fromByteArray(data: ByteArray, dtype: T): TensorData<T, V> {
        return when (dtype) {
            is FP32 -> {
                require(data.size % 4 == 0) { "Byte array size must be divisible by 4 for FP32" }
                val floats = FloatArray(data.size / 4) { i ->
                    val bytes = data.sliceArray(i * 4 until (i + 1) * 4)
                    // Convert little-endian bytes to float
                    val intBits = (bytes[3].toInt() and 0xFF shl 24) or
                            (bytes[2].toInt() and 0xFF shl 16) or
                            (bytes[1].toInt() and 0xFF shl 8) or
                            (bytes[0].toInt() and 0xFF)
                    Float.fromBits(intBits)
                }
                createFloatTensorData(Shape(floats.size), floats, dtype) as TensorData<T, V>
            }

            is Int8 -> {
                createByteTensorData<T>(Shape(data.size), data) as TensorData<T, V>
            }

            is Int4 -> {
                val values = mutableListOf<Byte>()
                for (byte in data) {
                    // Extract lower nibble (4 bits) and upper nibble
                    values.add((byte.toInt() and 0x0F).toByte()) // Lower nibble
                    values.add(((byte.toInt() and 0xF0) ushr 4).toByte()) // Upper nibble
                }
                createByteTensorData<T>(Shape(values.size), values.toByteArray()) as TensorData<T, V>
            }

            is Ternary -> {
                val values = mutableListOf<Byte>()
                for (byte in data) {
                    // Extract 2-bit values from byte (4 values per byte)
                    for (i in 0..3) {
                        val bits = (byte.toInt() shr (i * 2)) and 0x03
                        val value = when (bits) {
                            0b00 -> -1
                            0b01 -> 0
                            0b10 -> 1
                            else -> 0 // fallback for 0b11
                        }
                        values.add(value.toByte())
                    }
                }
                createByteTensorData<T>(Shape(values.size), values.toByteArray()) as TensorData<T, V>
            }

            else -> throw IllegalArgumentException("Unsupported dtype for fromByteArray: ${dtype.name}")
        }
    }

    /**
     * Creates a tensor from float array with specified data type (overloaded version).
     */
    @Suppress("UNCHECKED_CAST")
    public fun <T : DType, V> fromFloatArray(data: FloatArray, dtype: T): TensorData<T, V> {
        return when (dtype) {
            is FP32, FP16 -> {
                createFloatTensorData(Shape(data.size), data, dtype) as TensorData<T, V>
            }

            else -> throw IllegalArgumentException("fromFloatArray only supports floating point types: ${dtype.name}")
        }
    }

    /**
     * Creates a tensor from int array with specified data type.
     */
    @Suppress("UNCHECKED_CAST")
    public fun <T : DType, V> fromIntArray(data: IntArray, dtype: T): TensorData<T, V> {
        return when (dtype) {
            is Int32 -> {
                createIntTensorData(Shape(data.size), data) as TensorData<T, V>
            }

            else -> throw IllegalArgumentException("fromIntArray only supports integer types: ${dtype.name}")
        }
    }
}