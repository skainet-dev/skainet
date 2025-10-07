package sk.ai.net.core.tensor.data

import sk.ai.net.lang.tensor.Shape


/**
 * Computes standard row-major strides for the given shape.
 */
internal fun Shape.computeStrides(): IntArray {
    if (dimensions.isEmpty()) return intArrayOf()

    val strides = IntArray(dimensions.size)
    strides[dimensions.size - 1] = 1

    for (i in dimensions.size - 2 downTo 0) {
        strides[i] = strides[i + 1] * dimensions[i + 1]
    }

    return strides
}

