package sk.ainet.core.tensor.backend

import sk.ainet.core.tensor.*
import kotlin.math.*

/**
 * Generic CPU-based backend implementation for common tensor operations.
 *
 * This unified backend can work with any DType and value type combination,
 * providing common operations that work across different data types.
 * Type-specific optimizations and operations are handled by specialized backends.
 *
 * @param T The DType (data type) such as FP32, Int32, Int8, FP16, Int4, Ternary
 * @param V The value type such as Float, Int, Byte, etc.
 */
public abstract class CpuBackendGeneric<T : DType, V>(
    override val name: String
) : ComputeBackend<T, V, Tensor<T, V>> {

    /**
     * Provides type-specific conversion from Double to the value type V.
     * Must be implemented by concrete backends for each type.
     */
    protected abstract fun convertFromDouble(value: Double): V

    /**
     * Provides type-specific conversion from the value type V to Double.
     * Must be implemented by concrete backends for each type.
     */
    protected abstract fun convertToDouble(value: V): Double

    /**
     * Provides zero value for the specific type.
     * Must be implemented by concrete backends for each type.
     */
    protected abstract fun zeroValue(): V

    /**
     * Provides one value for the specific type.
     * Must be implemented by concrete backends for each type.
     */
    protected abstract fun oneValue(): V

    /**
     * Creates a tensor with the given shape and data.
     * Must be implemented by concrete backends to create the right tensor type.
     */
    protected abstract fun createTensor(shape: Shape, data: Array<V>): Tensor<T, V>

    /**
     * Creates an array of the value type V with the given size and initializer.
     * Must be implemented by concrete backends since V is not reified.
     */
    protected abstract fun createValueArray(size: Int, initializer: (Int) -> V): Array<V>

    // Basic tensor creation methods
    public fun zeros(shape: Shape): Tensor<T, V> {
        val data = createValueArray(shape.volume) { zeroValue() }
        return createTensor(shape, data)
    }

    public fun ones(shape: Shape): Tensor<T, V> {
        val data = createValueArray(shape.volume) { oneValue() }
        return createTensor(shape, data)
    }

    public fun full(shape: Shape, value: V): Tensor<T, V> {
        val data = createValueArray(shape.volume) { value }
        return createTensor(shape, data)
    }

    // Broadcasting utility functions
    protected fun canBroadcast(shape1: Shape, shape2: Shape): Boolean {
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

    protected fun getBroadcastShape(shape1: Shape, shape2: Shape): Shape {
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

    protected fun broadcastIndex(flatIndex: Int, resultShape: Shape, originalShape: Shape): Int {
        val resultRank = resultShape.rank
        val originalRank = originalShape.rank
        
        var index = 0
        var remaining = flatIndex
        
        for (i in 0 until resultRank) {
            val resultDim = resultShape[i]
            val originalDim = if (i >= resultRank - originalRank) originalShape[i - (resultRank - originalRank)] else 1
            
            val coord = remaining / (resultShape.volume / resultShape.cumulativeProduct(i + 1))
            remaining %= (resultShape.volume / resultShape.cumulativeProduct(i + 1))
            
            val originalCoord = if (originalDim == 1) 0 else coord
            index += originalCoord * (originalShape.volume / originalShape.cumulativeProduct(maxOf(0, i - (resultRank - originalRank)) + 1))
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

    // Generic matrix multiplication implementation
    override fun matmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        require(a.shape.rank == 2 && b.shape.rank == 2) { "Matrix multiplication requires 2D tensors" }
        require(a.shape[1] == b.shape[0]) { "Matrix dimensions don't match for multiplication" }

        val rows = a.shape[0]
        val cols = b.shape[1]
        val inner = a.shape[1]
        val result = createValueArray(rows * cols) { zeroValue() }

        val aData = createValueArray(a.shape.volume) { zeroValue() }
        val bData = createValueArray(b.shape.volume) { zeroValue() }
        a.copyTo(aData)
        b.copyTo(bData)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                var sum = 0.0
                for (k in 0 until inner) {
                    val aVal = convertToDouble(aData[i * inner + k])
                    val bVal = convertToDouble(bData[k * cols + j])
                    sum += aVal * bVal
                }
                result[i * cols + j] = convertFromDouble(sum)
            }
        }

        return createTensor(Shape(rows, cols), result)
    }

    // 4D matrix multiplication - basic implementation
    override fun matmul4d(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        throw UnsupportedOperationException("4D matrix multiplication not yet implemented in generic backend")
    }

    // Generic scaling operation
    override fun scale(a: Tensor<T, V>, scalar: Double): Tensor<T, V> {
        val result = createValueArray(a.shape.volume) { zeroValue() }
        val aData = createValueArray(a.shape.volume) { zeroValue() }
        a.copyTo(aData)

        for (i in aData.indices) {
            val value = convertToDouble(aData[i]) * scalar
            result[i] = convertFromDouble(value)
        }

        return createTensor(a.shape, result)
    }

    // Generic dot product
    override fun dot(a: Tensor<T, V>, b: Tensor<T, V>): Double {
        require(a.shape == b.shape) { "Tensors must have same shape for dot product" }

        val aData = createValueArray(a.shape.volume) { zeroValue() }
        val bData = createValueArray(b.shape.volume) { zeroValue() }
        a.copyTo(aData)
        b.copyTo(bData)

        var sum = 0.0
        for (i in aData.indices) {
            sum += convertToDouble(aData[i]) * convertToDouble(bData[i])
        }
        return sum
    }
}