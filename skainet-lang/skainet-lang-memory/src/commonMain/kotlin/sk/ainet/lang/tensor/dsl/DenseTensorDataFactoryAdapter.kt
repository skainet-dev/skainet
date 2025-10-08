package sk.ainet.lang.tensor.dsl

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.Int32
import sk.ainet.lang.types.Int8
import kotlin.random.Random

/**
 * Concrete implementation of TensorDataFactory that adapts the existing DenseTensorDataFactory
 * to work with the tensor DSL
 */
public class DenseTensorDataFactoryAdapter<T : DType, V> : TensorDataFactory<T, V> {
    
    private val denseFactory = DenseTensorDataFactory()
    
    override fun zeros(shape: Shape, dtype: T): TensorData<T, V> {
        @Suppress("UNCHECKED_CAST")
        return when (dtype) {
            is FP32 -> denseFactory.full(shape, 0.0f, dtype) as TensorData<T, V>
            is FP16 -> denseFactory.full(shape, 0.0f, dtype) as TensorData<T, V>
            is Int32 -> denseFactory.full(shape, 0, dtype) as TensorData<T, V>
            is Int8 -> denseFactory.full(shape, 0.toByte(), dtype) as TensorData<T, V>
            else -> denseFactory.full(shape, 0.0f, dtype) as TensorData<T, V>
        }
    }
    
    override fun ones(shape: Shape, dtype: T): TensorData<T, V> {
        @Suppress("UNCHECKED_CAST")
        return when (dtype) {
            is FP32 -> denseFactory.full(shape, 1.0f, dtype) as TensorData<T, V>
            is FP16 -> denseFactory.full(shape, 1.0f, dtype) as TensorData<T, V>
            is Int32 -> denseFactory.full(shape, 1, dtype) as TensorData<T, V>
            is Int8 -> denseFactory.full(shape, 1.toByte(), dtype) as TensorData<T, V>
            else -> denseFactory.full(shape, 1.0f, dtype) as TensorData<T, V>
        }
    }
    
    override fun full(shape: Shape, value: Number, dtype: T): TensorData<T, V> {
        @Suppress("UNCHECKED_CAST")
        return denseFactory.full(shape, value, dtype) as TensorData<T, V>
    }
    
    override fun randn(shape: Shape, mean: Float, std: Float, dtype: T, random: Random): TensorData<T, V> {
        @Suppress("UNCHECKED_CAST")
        return denseFactory.randn(shape, mean, std, dtype, random) as TensorData<T, V>
    }
    
    override fun uniform(shape: Shape, min: Float, max: Float, dtype: T, random: Random): TensorData<T, V> {
        @Suppress("UNCHECKED_CAST")
        return when (dtype) {
            is FP32 -> {
                val data = FloatArray(shape.volume) { random.nextFloat() * (max - min) + min }
                denseFactory.fromFloatArray(shape, data, dtype) as TensorData<T, V>
            }
            is FP16 -> {
                val data = FloatArray(shape.volume) { random.nextFloat() * (max - min) + min }
                denseFactory.fromFloatArray(shape, data, dtype) as TensorData<T, V>
            }
            is Int32 -> {
                val data = IntArray(shape.volume) { 
                    (random.nextFloat() * (max - min) + min).toInt() 
                }
                denseFactory.fromIntArray<T, V>(data, dtype)
            }
            is Int8 -> {
                val data = ByteArray(shape.volume) { 
                    (random.nextFloat() * (max - min) + min).toInt().toByte()
                }
                denseFactory.fromByteArray<T, V>(data, dtype)
            }
            else -> {
                val data = FloatArray(shape.volume) { random.nextFloat() * (max - min) + min }
                denseFactory.fromFloatArray(shape, data, dtype) as TensorData<T, V>
            }
        }
    }
    
    override fun init(shape: Shape, dtype: T, generator: (indices: IntArray) -> V): TensorData<T, V> {
        @Suppress("UNCHECKED_CAST")
        return when (dtype) {
            is FP32 -> {
                val data = FloatArray(shape.volume) { index ->
                    val indices = computeIndicesFromFlat(index, shape)
                    (generator(indices) as Float)
                }
                denseFactory.fromFloatArray(shape, data, dtype) as TensorData<T, V>
            }
            is FP16 -> {
                val data = FloatArray(shape.volume) { index ->
                    val indices = computeIndicesFromFlat(index, shape)
                    (generator(indices) as Float)
                }
                denseFactory.fromFloatArray(shape, data, dtype) as TensorData<T, V>
            }
            is Int32 -> {
                val data = IntArray(shape.volume) { index ->
                    val indices = computeIndicesFromFlat(index, shape)
                    (generator(indices) as Int)
                }
                denseFactory.fromIntArray(data, dtype) as TensorData<T, V>
            }
            is Int8 -> {
                val data = ByteArray(shape.volume) { index ->
                    val indices = computeIndicesFromFlat(index, shape)
                    (generator(indices) as Byte)
                }
                denseFactory.fromByteArray(data, dtype) as TensorData<T, V>
            }
            else -> {
                val data = FloatArray(shape.volume) { index ->
                    val indices = computeIndicesFromFlat(index, shape)
                    (generator(indices) as Float)
                }
                denseFactory.fromFloatArray(shape, data, dtype) as TensorData<T, V>
            }
        }
    }
    
    override fun randomInit(shape: Shape, dtype: T, generator: (random: Random) -> V, random: Random): TensorData<T, V> {
        @Suppress("UNCHECKED_CAST")
        return when (dtype) {
            is FP32 -> {
                val data = FloatArray(shape.volume) { 
                    generator(random) as Float
                }
                denseFactory.fromFloatArray(shape, data, dtype) as TensorData<T, V>
            }
            is FP16 -> {
                val data = FloatArray(shape.volume) { 
                    generator(random) as Float
                }
                denseFactory.fromFloatArray(shape, data, dtype) as TensorData<T, V>
            }
            is Int32 -> {
                val data = IntArray(shape.volume) { 
                    generator(random) as Int
                }
                denseFactory.fromIntArray(data, dtype) as TensorData<T, V>
            }
            is Int8 -> {
                val data = ByteArray(shape.volume) { 
                    generator(random) as Byte
                }
                denseFactory.fromByteArray(data, dtype) as TensorData<T, V>
            }
            else -> {
                val data = FloatArray(shape.volume) { 
                    generator(random) as Float
                }
                denseFactory.fromFloatArray(shape, data, dtype) as TensorData<T, V>
            }
        }
    }
    
    /**
     * Converts a flat index back to multidimensional indices for custom initialization
     */
    private fun computeIndicesFromFlat(flatIndex: Int, shape: Shape): IntArray {
        val indices = IntArray(shape.rank)
        var remaining = flatIndex
        
        for (i in shape.rank - 1 downTo 0) {
            indices[i] = remaining % shape[i]
            remaining /= shape[i]
        }
        
        return indices
    }
}

/**
 * Convenience function to create a DenseTensorDataFactoryAdapter
 */
public fun <T : DType, V> denseTensorFactory(): TensorDataFactory<T, V> = 
    DenseTensorDataFactoryAdapter()

/**
 * Extension function to build tensor with default dense factory
 */
public fun <T : DType, V> TensorInitializer<T, V>.buildDense() = 
    build(denseTensorFactory<T, V>())