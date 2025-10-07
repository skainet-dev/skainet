package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import sk.ainet.lang.types.Int4
import sk.ainet.lang.types.Int8
import sk.ainet.lang.types.Ternary
import kotlin.jvm.JvmName

public class DenseTensorDataFactory {
    public fun from(value: Int): TensorData<
            Int32, Int> {
        return object : TensorData<Int32, Int> {
            override val shape: Shape
                get() = Shape.Companion(1)

            override fun get(vararg indices: Int): Int = value
        }
    }

    public fun from(value: Float): TensorData<FP32, Float> {
        return object : TensorData<FP32, Float> {
            override val shape: Shape
                get() = Shape.Companion(1)

            override fun get(vararg indices: Int): Float = value
        }
    }

    @JvmName("vectorFromInt")
    public fun fromArray(arrayOf: Array<Int>): TensorData<Int32, Int> {
        class IntTensorData(private val data: IntArray) : TensorData<Int32, Int> {
            override val shape: Shape
                get() = Shape.Companion(data.size)

            override fun get(vararg indices: Int): Int = data[indices[0]]
        }
        return IntTensorData(arrayOf.toIntArray())
    }

    @JvmName("vectorFromFloat")
    public fun fromArray(arrayOf: Array<Float>): TensorData<FP32, Float> {
        class FloatTensorData(private val data: FloatArray) : TensorData<FP32, Float> {
            override val shape: Shape
                get() = Shape.Companion(data.size)

            override fun get(vararg indices: Int): Float = data[indices[0]]
        }
        return FloatTensorData(arrayOf.toFloatArray())
    }

    @Suppress("UNCHECKED_CAST")
    public fun <T : DType, V> fromFloatArray(
        data: FloatArray,
        dtype: T
    ): TensorData<T, V> {
        return when (dtype) {
            is FP32 -> {
                class FP32FloatTensorData() : TensorData<FP32, Float> {
                    override val shape: Shape
                        get() = Shape.Companion(data.size)

                    override fun get(vararg indices: Int): Float = data[indices[0]]
                }
                FP32FloatTensorData() as TensorData<T, V>
            }
            is FP16 -> {
                class FP16FloatTensorData() : TensorData<FP16, Float> {
                    override val shape: Shape
                        get() = Shape.Companion(data.size)

                    override fun get(vararg indices: Int): Float = data[indices[0]]
                }
                FP16FloatTensorData() as TensorData<T, V>
            }
            else -> throw IllegalArgumentException("Unsupported dtype for FloatArray: ${dtype.name}")
        }
    }

    @Suppress("UNCHECKED_CAST")
    public fun <T : DType, V> fromIntArray(
        data: IntArray,
        dtype: T
    ): TensorData<T, V> {
        return when (dtype) {
            is Int32 -> {
                class Int32IntTensorData() : TensorData<Int32, Int> {
                    override val shape: Shape
                        get() = Shape.Companion(data.size)

                    override fun get(vararg indices: Int): Int = data[indices[0]]
                }
                Int32IntTensorData() as TensorData<T, V>
            }
            else -> throw IllegalArgumentException("Unsupported dtype for IntArray: ${dtype.name}")
        }
    }

    @Suppress("UNCHECKED_CAST")
    public fun <T : DType, V> fromByteArray(
        bytes: ByteArray,
        dtype: T
    ): TensorData<T, V> {
        return when (dtype) {
            is FP32 -> {
                class FP32ByteTensorData(private val data: ByteArray) : TensorData<FP32, Float> {
                    override val shape: Shape
                        get() = Shape.Companion(data.size / 4) // 4 bytes per float

                    override fun get(vararg indices: Int): Float {
                        val index = indices[0] * 4
                        return Float.fromBits(
                            (data[index].toInt() and 0xFF) or
                            ((data[index + 1].toInt() and 0xFF) shl 8) or
                            ((data[index + 2].toInt() and 0xFF) shl 16) or
                            ((data[index + 3].toInt() and 0xFF) shl 24)
                        )
                    }
                }
                FP32ByteTensorData(bytes) as TensorData<T, V>
            }
            is Int8 -> {
                class Int8ByteTensorData(private val data: ByteArray) : TensorData<Int8, Byte> {
                    override val shape: Shape
                        get() = Shape.Companion(data.size)

                    override fun get(vararg indices: Int): Byte = data[indices[0]]
                }
                Int8ByteTensorData(bytes) as TensorData<T, V>
            }
            is Int4 -> {
                class Int4ByteTensorData(private val data: ByteArray) : TensorData<Int4, Byte> {
                    override val shape: Shape
                        get() = Shape.Companion(data.size * 2) // 2 int4 values per byte

                    override fun get(vararg indices: Int): Byte {
                        val byteIndex = indices[0] / 2
                        val isLowerNibble = indices[0] % 2 == 0
                        val byte = data[byteIndex].toInt()
                        return if (isLowerNibble) {
                            (byte and 0x0F).toByte() // Lower 4 bits
                        } else {
                            ((byte and 0xF0) shr 4).toByte() // Upper 4 bits
                        }
                    }
                }
                Int4ByteTensorData(bytes) as TensorData<T, V>
            }
            is Ternary -> {
                class TernaryByteTensorData(private val data: ByteArray) : TensorData<Ternary, Byte> {
                    override val shape: Shape
                        get() = Shape.Companion(data.size * 4) // 4 ternary values per byte (2 bits each)

                    override fun get(vararg indices: Int): Byte {
                        val byteIndex = indices[0] / 4
                        val bitPosition = (indices[0] % 4) * 2
                        val byte = data[byteIndex].toInt()
                        val value = (byte shr bitPosition) and 0x03 // Extract 2 bits
                        // Convert 2-bit value to ternary (-1, 0, 1)
                        return when (value) {
                            0 -> (-1).toByte()
                            1 -> 0.toByte()
                            2 -> 1.toByte()
                            else -> 0.toByte() // Default fallback
                        }
                    }
                }
                TernaryByteTensorData(bytes) as TensorData<T, V>
            }
            else -> throw IllegalArgumentException("Unsupported dtype: ${dtype.name}")
        }
    }

}