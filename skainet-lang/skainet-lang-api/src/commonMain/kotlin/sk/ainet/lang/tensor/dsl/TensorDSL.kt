package sk.ainet.lang.tensor.dsl

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.ops.SimpleTensorOperation
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.random.Random

/**
 * DSL for creating tensors with various initialization strategies.
 * 
 * This DSL provides a fluent interface for tensor creation, supporting:
 * - MLP use cases (weights, bias)
 * - Image processing (BCHW format)
 * - Various initializers: zeros, ones, random distributions
 * - Custom initialization functions
 */

/**
 * Entry point for tensor DSL - creates a tensor builder
 */
public fun <T : DType, V> tensor(dtype: T): TensorBuilder<T, V> = TensorBuilder(dtype)


/**
 * Builder class for constructing tensors with various initialization strategies
 */
public class TensorBuilder<T : DType, V>(private val dtype: T) {
    
    /**
     * Creates a tensor with specified shape and initialization strategy
     */
    public fun shape(vararg dimensions: Int): ShapeBuilder<T, V> = 
        ShapeBuilder(dtype, Shape(*dimensions))
    
    /**
     * Creates a tensor with specified shape
     */
    public fun shape(shape: Shape): ShapeBuilder<T, V> = 
        ShapeBuilder(dtype, shape)
}

/**
 * Builder for tensors with a defined shape, allowing initialization
 */
public class ShapeBuilder<T : DType, V>(
    private val dtype: T, 
    private val shape: Shape
) {
    
    /**
     * Initialize tensor with zeros
     */
    public fun zeros(): TensorInitializer<T, V> = 
        TensorInitializer(dtype, shape, InitializationType.Zeros)
    
    /**
     * Initialize tensor with ones
     */
    public fun ones(): TensorInitializer<T, V> = 
        TensorInitializer(dtype, shape, InitializationType.Ones)
    
    /**
     * Initialize tensor with a constant value
     */
    public fun fill(value: Number): TensorInitializer<T, V> = 
        TensorInitializer(dtype, shape, InitializationType.Fill(value))
    
    /**
     * Initialize tensor with normal distribution
     */
    public fun randn(mean: Float = 0.0f, std: Float = 1.0f, random: Random = Random.Default): TensorInitializer<T, V> = 
        TensorInitializer(dtype, shape, InitializationType.Normal(mean, std, random))
    
    /**
     * Initialize tensor with uniform distribution
     */
    public fun uniform(min: Float = 0.0f, max: Float = 1.0f, random: Random = Random.Default): TensorInitializer<T, V> = 
        TensorInitializer(dtype, shape, InitializationType.Uniform(min, max, random))
    
    /**
     * Initialize tensor with custom function
     */
    public fun init(generator: (indices: IntArray) -> V): TensorInitializer<T, V> = 
        TensorInitializer(dtype, shape, InitializationType.Custom(generator))
    
    /**
     * Initialize tensor with custom random distribution
     */
    public fun randomInit(generator: (random: Random) -> V, random: Random = Random.Default): TensorInitializer<T, V> = 
        TensorInitializer(dtype, shape, InitializationType.RandomCustom(generator, random))
}

/**
 * Initializer that holds the complete tensor specification
 */
public class TensorInitializer<T : DType, V>(
    private val dtype: T,
    private val shape: Shape,
    private val initType: InitializationType<V>
) {
    
    /**
     * Build the actual tensor with the specified initialization
     */
    public fun build(factory: TensorDataFactory<T, V>): Tensor<T, V> {
        val data = when (initType) {
            is InitializationType.Zeros -> factory.zeros(shape, dtype)
            is InitializationType.Ones -> factory.ones(shape, dtype)
            is InitializationType.Fill -> factory.full(shape, initType.value, dtype)
            is InitializationType.Normal -> factory.randn(shape, initType.mean, initType.std, dtype, initType.random)
            is InitializationType.Uniform -> factory.uniform(shape, initType.min, initType.max, dtype, initType.random)
            is InitializationType.Custom -> factory.init(shape, dtype, initType.generator)
            is InitializationType.RandomCustom -> factory.randomInit(shape, dtype, initType.generator, initType.random)
        }
        
        return SimpleTensor(data, SimpleTensorOperation(), dtype)
    }
}

/**
 * Types of tensor initialization
 */
public sealed class InitializationType<out V> {
    public object Zeros : InitializationType<Nothing>()
    public object Ones : InitializationType<Nothing>()
    public data class Fill<V>(val value: Number) : InitializationType<V>()
    public data class Normal<V>(val mean: Float, val std: Float, val random: Random) : InitializationType<V>()
    public data class Uniform<V>(val min: Float, val max: Float, val random: Random) : InitializationType<V>()
    public data class Custom<V>(val generator: (indices: IntArray) -> V) : InitializationType<V>()
    public data class RandomCustom<V>(val generator: (random: Random) -> V, val random: Random) : InitializationType<V>()
}

/**
 * Interface for tensor data factory used by the DSL
 */
public interface TensorDataFactory<T : DType, V> {
    public fun zeros(shape: Shape, dtype: T): TensorData<T, V>
    public fun ones(shape: Shape, dtype: T): TensorData<T, V>
    public fun full(shape: Shape, value: Number, dtype: T): TensorData<T, V>
    public fun randn(shape: Shape, mean: Float, std: Float, dtype: T, random: Random): TensorData<T, V>
    public fun uniform(shape: Shape, min: Float, max: Float, dtype: T, random: Random): TensorData<T, V>
    public fun init(shape: Shape, dtype: T, generator: (indices: IntArray) -> V): TensorData<T, V>
    public fun randomInit(shape: Shape, dtype: T, generator: (random: Random) -> V, random: Random): TensorData<T, V>
}

/**
 * Simple tensor implementation for the DSL
 */
private class SimpleTensor<T : DType, V>(
    override val data: TensorData<T, V>,
    override val ops: SimpleTensorOperation<V>,
    override val dtype: T
) : Tensor<T, V>