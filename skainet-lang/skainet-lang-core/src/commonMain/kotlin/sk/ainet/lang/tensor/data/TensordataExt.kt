package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType


/**
 * Computes standard row-major strides for the given shape.
 *
 * https://en.wikipedia.org/wiki/Row-_and_column-major_order
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

/**
 * Extension property to provide TensorData factory methods within DenseTensorDataFactory context
 */
//public val DenseTensorDataFactory.TensorData: TensorDataFactory get() = TensorDataFactory(this)

/*
public class TensorDataFactory(public val factory: DenseTensorDataFactory) {
    public fun <T : DType, V> scalar(value: V): TensorData<T, V> {
        return factory.scalar(value)
    }

    public fun <T : DType, V> vector(values: Array<V>): TensorData<T, V> {
        return factory.vector(values)
    }

    public fun <T : DType, V> matrix(vararg rows: Array<V>): TensorData<T, V> {
        return factory.matrix(*rows)
    }
}

 */

