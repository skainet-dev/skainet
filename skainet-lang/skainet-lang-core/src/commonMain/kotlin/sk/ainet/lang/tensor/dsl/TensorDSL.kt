package sk.ainet.lang.tensor.dsl

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.DType
import kotlin.random.Random
import kotlin.reflect.KClass

// DSL Marker to restrict the DSL to its intended scope
@DslMarker
public annotation class TensorDsl

@TensorDsl
public interface TensorContextDslItem

@TensorDsl
public fun <T : DType, V> tensor(
    dataFactory: TensorDataFactory = DenseTensorDataFactory(),
    ops: TensorOps<V> = VoidTensorOps(),
    dtype: KClass<T>,
    content: TensorDefineDsl<T, V>.() -> Tensor<T, V>
): Tensor<T, V> {
    val dsl = TensorDefineDslImpl(dtype = dtype, tensorDataFactory = dataFactory, ops = ops)
    return dsl.content()
}


@TensorDsl
public interface TensorDefineDsl<T : DType, V> : TensorContextDslItem {
    public fun tensor(content: TensorFactoryContext<T, V>.() -> Tensor<T, V>): Tensor<T, V>
}

internal class TensorDefineDslImpl<T : DType, V>(
    private val dtype: KClass<T>,
    private val tensorDataFactory: TensorDataFactory,
    private val ops: TensorOps<V>,
) : TensorDefineDsl<T, V> {

    /*
    override fun <T : DType> tensor(
        dType: DType,
        content: TensorFactoryContext<T, V>.() -> Tensor<T, V>
    ): Tensor<T, V> =
        sk.ainet.lang.tensor.dsl.tensor<T, V>(
            dataFactory = tensorDataFactory,
            ops = ops,
            dtype = dType::class,
            content = content
        )

     */

    fun create(): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    override fun tensor(content: TensorFactoryContext<T, V>.() -> Tensor<T, V>): Tensor<T, V> {
        val context = TensorFactoryContext<T, V>(tensorDataFactory, dtype)
        return context.content()
    }
}


public interface TensorCreationScope<T : DType, V> {
    public val factory: TensorDataFactory
    public val shape: Shape
    public val dtype: KClass<T>

    /**
     * Create tensor filled with zeros
     */
    public fun zeros(): Tensor<T, V> {
        val data = factory.zeros<T, V>(shape, dtype)
        return VoidOpsTensor(data, dtype)
    }

    /**
     * Create tensor filled with ones
     */
    public fun ones(): Tensor<T, V> {
        val data = factory.ones<T, V>(shape, dtype)
        return VoidOpsTensor(data, dtype)
    }

    /**
     * Create tensor filled with a constant value
     */
    public fun full(value: Number): Tensor<T, V> {
        val data = factory.full<T, V>(shape, dtype, value)
        return VoidOpsTensor(data, dtype)
    }

    /**
     * Create tensor with normal distribution
     */
    public fun randN(mean: Float = 0.0f, std: Float = 1.0f, random: Random = Random.Default): Tensor<T, V> {
        val data = factory.randn<T, V>(shape, dtype, mean, std, random)
        return VoidOpsTensor(data, dtype)
    }

    /**
     * Create tensor with uniform distribution
     */
    public fun uniform(min: Float = 0.0f, max: Float = 1.0f, random: Random = Random.Default): Tensor<T, V> {
        val data = factory.uniform<T, V>(shape, dtype, min, max, random)
        return VoidOpsTensor(data, dtype)
    }

    /**
     * Create tensor with custom initialization function
     */
    public fun init(generator: (indices: IntArray) -> V): Tensor<T, V> {
        val data = factory.init(shape, dtype, generator)
        return VoidOpsTensor(data, dtype)
    }

    /**
     * Create tensor with custom random initialization
     */
    public fun randomInit(generator: (random: Random) -> V, random: Random = Random.Default): Tensor<T, V> {
        val data = factory.randomInit(shape, dtype, generator, random)
        return VoidOpsTensor(data, dtype)
    }
}

/**
 * Implementation of TensorCreationScope
 */
public class TensorCreationScopeImpl<T : DType, V>(
    override val factory: TensorDataFactory,
    override val shape: Shape,
    override val dtype: KClass<T>
) : TensorCreationScope<T, V>

/**
 * Context wrapper that provides tensor data factory access
 */
public class TensorFactoryContext<T : DType, V>(
    private val factory: TensorDataFactory,
    private val dtype: KClass<T>
) {
    /**
     * Create tensor with specified shape and initialization strategy
     */
    public fun shape(vararg dimensions: Int, init: TensorCreationScope<T, V>.(Shape) -> Tensor<T, V>): Tensor<T, V> {
        return shape(Shape(*dimensions), init)
    }

    /**
     * Create tensor with specified shape and initialization strategy
     */
    public fun shape(shape: Shape, init: TensorCreationScope<T, V>.(Shape) -> Tensor<T, V>): Tensor<T, V> {
        val scope = TensorCreationScopeImpl<T, V>(factory, shape, dtype)
        return scope.init(shape)
    }
}

/**
 * Entry point for context-aware tensor DSL
 */
/*
public fun <T : DType, V> tensor(
    factory: TensorDataFactory,
    content: TensorFactoryContext<T, V>.() -> Tensor<T, V>
): Tensor<T, V> {
    val context = TensorFactoryContext<T, V>(factory, T::class)
    return context.content()
}

 */

/*
 public inline fun <reified T : DType, V> tensor(
    content: TensorFactoryContext<T, V>.() -> Tensor<T, V>
): Tensor<T, V> {
    // If an execution context is active, use its factory and bind ops
    val ctx = sk.ainet.context.ExecutionEnvironment.current
    return if (ctx != null) {
        @Suppress("UNCHECKED_CAST")
        val typed = ctx as sk.ainet.context.ExecutionContext<V>
        val t = tensor<T, V>(typed.tensorDataFactory, content)
        t.withOps(typed.ops)
    } else {
        tensor(DenseTensorDataFactory(), content)
    }
}

 */


/**
 * Builder class for constructing tensors with various initialization strategies
 */
public class TensorBuilder<T : DType, V>(private val dtype: KClass<T>) {

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
    private val dtype: KClass<T>,
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
    private val dtype: KClass<T>,
    private val shape: Shape,
    private val initType: InitializationType<V>
) {

    /**
     * Build the actual tensor with the specified initialization
     */
    public fun build(factory: TensorDataFactory): Tensor<T, V> {
        val data = when (initType) {
            is InitializationType.Zeros -> factory.zeros(shape, dtype)
            is InitializationType.Ones -> factory.ones(shape, dtype)
            is InitializationType.Fill -> factory.full(shape, dtype, initType.value)
            is InitializationType.Normal -> factory.randn(
                shape,
                dtype,
                initType.mean,
                initType.std,
                initType.random
            )

            is InitializationType.Uniform -> factory.uniform(
                shape,
                dtype,
                initType.min,
                initType.max,
                initType.random
            )

            is InitializationType.Custom -> factory.init(shape, dtype, initType.generator)
            is InitializationType.RandomCustom -> factory.randomInit(
                shape,
                dtype,
                initType.generator,
                initType.random
            )
        }

        return VoidOpsTensor(data, dtype)
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
    public data class RandomCustom<V>(val generator: (random: Random) -> V, val random: Random) :
        InitializationType<V>()
}