package sk.ainet.lang.tensor.data.dense

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.ItemsAccessor
import sk.ainet.lang.tensor.data.computeStrides

/**
 * Dense memory-backed tensor array for Int4 (4-bit signed integer) data.
 *
 * Stores two Int4 values per byte:
 *  - High nibble: bits 4–7
 *  - Low nibble: bits 0–3
 *
 * Values range from -8 to +7.
 */
public class DenseInt4TensorArray(
    public val shape: Shape,
    private val data: ByteArray,
    private val offset: Int = 0
) : ItemsAccessor<Int> {

    public constructor(dimensions: List<Int>, data: ByteArray) : this(Shape(dimensions.toIntArray()), data)

    private val strides: IntArray = shape.computeStrides()

    override fun get(vararg indices: Int): Int {
        val flatIndex = offset + calcFlatIndex(shape, strides, *indices)
        val byteIndex = flatIndex / 2
        val isHighNibble = flatIndex % 2 == 1
        val byteValue = data[byteIndex].toInt()

        val nibble = if (isHighNibble) {
            (byteValue ushr 4) and 0x0F
        } else {
            byteValue and 0x0F
        }

        // Convert to signed Int4 (-8..7)
        return if (nibble >= 8) nibble - 16 else nibble
    }

    override fun set(vararg indices: Int, value: Int) {
        require(value in -8..7) { "Int4 value must be in range -8..7, but got $value" }

        val flatIndex = offset + calcFlatIndex(shape, strides, *indices)
        val byteIndex = flatIndex / 2
        val isHighNibble = flatIndex % 2 == 1

        var byteValue = data[byteIndex].toInt() and 0xFF
        val nibble = (if (value < 0) value + 16 else value) and 0x0F

        byteValue = if (isHighNibble) {
            (byteValue and 0x0F) or (nibble shl 4)
        } else {
            (byteValue and 0xF0) or nibble
        }

        data[byteIndex] = byteValue.toByte()
    }
}