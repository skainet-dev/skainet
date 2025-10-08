package sk.ainet.lang.tensor.memory

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.ItemsAccessor
import sk.ainet.lang.tensor.data.computeStrides

/**
 * Stores ternary data (-1, 0, 1) compactly using 2 bits per value.
 *
 *  Encoding scheme (2 bits per value):
 *  - 00 -> 0
 *  - 01 -> 1
 *  - 10 -> -1
 *  - 11 -> reserved (unused)
 */
public class DenseTernaryTensorArray(
    public val shape: Shape,
    private val data: ByteArray,
    private val offset: Int = 0,
) : ItemsAccessor<Byte> {

    public constructor(dimensions: List<Int>, data: ByteArray)
            : this(Shape(dimensions.toIntArray()), data)

    private val strides: IntArray = shape.computeStrides()

    private fun getBitOffset(flatIndex: Int): Int = flatIndex * 2
    private fun getByteIndex(flatIndex: Int): Int = (getBitOffset(flatIndex) / 8) + offset
    private fun getBitPosition(flatIndex: Int): Int = getBitOffset(flatIndex) % 8

    override fun get(vararg indices: Int): Byte {
        val flatIndex = calcFlatIndex(shape, strides, *indices)
        val byteIndex = getByteIndex(flatIndex)
        val bitPos = getBitPosition(flatIndex)

        val rawBits = ((data[byteIndex].toInt() ushr bitPos) and 0b11)

        return when (rawBits) {
            0b00 -> 0
            0b01 -> 1
            0b10 -> -1
            else -> 0 // reserved, treat as 0 for safety
        }.toByte()
    }

    override fun set(vararg indices: Int, value: Byte) {
        val flatIndex = calcFlatIndex(shape, strides, *indices)
        val byteIndex = getByteIndex(flatIndex)
        val bitPos = getBitPosition(flatIndex)

        val encodedBits = when (value.toInt()) {
            0 -> 0b00
            1 -> 0b01
            -1 -> 0b10
            else -> throw IllegalArgumentException("Value must be -1, 0, or 1, got $value")
        }

        // Clear existing 2 bits
        data[byteIndex] = (data[byteIndex].toInt() and (0b11 shl bitPos).inv()).toByte()

        // Set new bits
        data[byteIndex] = (data[byteIndex].toInt() or (encodedBits shl bitPos)).toByte()
    }

    public companion object Companion {
        /**
         * Utility to compute how many bytes are needed for a given number of ternary elements.
         */
        public fun requiredBytesForElements(count: Int): Int = (count * 2 + 7) / 8
    }
}