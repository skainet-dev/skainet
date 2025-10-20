package sk.ainet.lang.tensor.data.dense

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.ItemsAccessor
import sk.ainet.lang.tensor.data.computeStrides

internal fun calcFlatIndex(shape: Shape, strides: IntArray, vararg indices: Int): Int {
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

// DoubleArray row-major
public class DenseFloatTensorArray(
    public val shape: Shape,
    private val data: FloatArray,
    private val offset: Int = 0,
) : ItemsAccessor<Float> {
    public constructor(dimensions: List<Int>, data: FloatArray) : this(Shape(dimensions.toIntArray()), data)

    private val strides: IntArray = shape.computeStrides()

    override fun get(vararg indices: Int): Float {
        val flatIndex = offset + calcFlatIndex(shape, strides, *indices)
        return data[flatIndex]
    }

    override fun set(vararg indices: Int, value: Float) {
        val flatIndex = offset + calcFlatIndex(shape, strides, *indices)
        data[flatIndex] = value
    }
}