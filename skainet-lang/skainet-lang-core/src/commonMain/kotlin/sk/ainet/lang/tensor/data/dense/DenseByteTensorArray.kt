package sk.ainet.lang.tensor.data.dense

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.ItemsAccessor
import sk.ainet.lang.tensor.data.computeStrides

public class DenseByteTensorArray(
    public val shape: Shape,
    private val data: ByteArray,
    private val offset: Int = 0,
) : ItemsAccessor<Byte> {
    public constructor(dimensions: List<Int>, data: ByteArray) : this(Shape(dimensions.toIntArray()), data)

    private val strides: IntArray = shape.computeStrides()

    override fun get(vararg indices: Int): Byte {
        val flatIndex = offset + calcFlatIndex(shape, strides, *indices)
        return data[flatIndex]
    }

    override fun set(vararg indices: Int, value: Byte) {
        val flatIndex = offset + calcFlatIndex(shape, strides, *indices)
        data[flatIndex] = value
    }
}