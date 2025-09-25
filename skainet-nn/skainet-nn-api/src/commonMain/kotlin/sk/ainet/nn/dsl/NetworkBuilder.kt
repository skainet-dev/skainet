package sk.ainet.nn.dsl

import kotlin.jvm.JvmName
import sk.ainet.nn.activations.ActivationsWrapperModule
import sk.ainet.core.tensor.Shape
import sk.ainet.core.tensor.Tensor
import sk.ainet.core.tensor.DType
import sk.ainet.core.tensor.FP32
import sk.ainet.core.tensor.Int8
import sk.ainet.core.tensor.Int32
import sk.ainet.core.tensor.TensorFactory
import sk.ainet.core.tensor.DefaultTensorFactories
import sk.ainet.core.tensor.backend.CpuBackend
import sk.ainet.nn.Flatten
import sk.ainet.nn.Input
import sk.ainet.nn.Linear
import sk.ainet.nn.Module
import sk.ainet.nn.topology.MLP


// DSL Marker to restrict the DSL to its intended scope
@DslMarker
public annotation class NetworkDsl

/**
 * Generic network builder function that creates a neural network with specified data type and value type.
 *
 * @param T The data type (DType) - must extend DType (e.g., FP32, FP16, Int8, Int32, Ternary, Int4)
 * @param V The value type - must match the DType's native type:
 *   - FP32 → Float
 *   - FP16 → Float (promoted)
 *   - Int32 → Int
 *   - Int8 → Byte
 *   - Int4 → Byte (promoted)
 *   - Ternary → Byte (special case)
 * @param content The DSL content block that defines the network structure
 * @return A Module<T, V> representing the complete neural network
 *
 * Example usage:
 * ```kotlin
 * val fpNetwork = network<FP32, Float> {
 *     input(784)
 *     dense(128) { weights { shape -> CpuTensorFP32.random(shape) } }
 *     dense(10) { weights { shape -> CpuTensorFP32.random(shape) } }
 * }
 *
 * val intNetwork = network<Int8, Byte> {
 *     input(28)
 *     dense(16) { weights { shape -> CpuTensorInt8.ones(shape) } }
 * }
 * ```
 */
@NetworkDsl
public fun <T : DType, V> network(
    factory: TensorFactory<T, V>,
    content: NeuralNetworkDsl<T, V>.() -> Unit
): Module<T, V> =
    NeuralNetworkDslImpl(factory)
        .apply(content)
        .create()

/**
 * Backward compatibility function - creates a network using FP32/Float precision.
 * This function maintains compatibility with existing code that doesn't specify generic types.
 * Uses CpuBackend as default factory.
 *
 * @param content The DSL content block that defines the network structure
 * @return A Module<FP32, Float> representing the complete neural network
 */
@NetworkDsl
@JvmName("networkFP32Default")
public fun network(content: NeuralNetworkDsl<FP32, Float>.() -> Unit): Module<FP32, Float> =
    network(CpuBackend(), content)

/**
 * Convenience function for creating FP32/Float precision networks.
 * Provides explicit type specification for better code readability.
 * Uses CpuBackend as default factory.
 *
 * @param content The DSL content block that defines the network structure
 * @return A Module<FP32, Float> representing the complete neural network
 */
@NetworkDsl
public fun networkFP32(content: NeuralNetworkDsl<FP32, Float>.() -> Unit): Module<FP32, Float> =
    network(CpuBackend(), content)

/**
 * Generic network builder function with automatic factory resolution.
 * This function automatically selects the appropriate TensorFactory based on the generic types.
 *
 * Currently supports:
 * - FP32, Float → Uses CPU FP32 backend
 * - Int8, Byte → Uses CPU Int8 backend
 * - Int32, Int → Uses CPU Int32 backend
 *
 * @param T The data type (DType) - must extend DType (e.g., FP32, Int8, Int32)
 * @param V The value type - must match the DType's native type
 * @param content The DSL content block that defines the network structure
 * @return A Module<T, V> representing the complete neural network
 *
 * Example usage:
 * ```kotlin
 * val fpNetwork = network<FP32, Float> {
 *     input(784)
 *     dense(128)
 *     dense(10)
 * }
 *
 * val intNetwork = network<Int8, Byte> {
 *     input(28)
 *     dense(16)
 * }
 * ```
 */
@NetworkDsl
@JvmName("networkWithAutoFactory")
public inline fun <reified T : DType, reified V> network(
    noinline content: NeuralNetworkDsl<T, V>.() -> Unit
): Module<T, V> {
    CpuBackend()
    val factory = when {
        T::class == FP32::class && V::class == Float::class -> {
            @Suppress("UNCHECKED_CAST")
            DefaultTensorFactories.getFP32Factory() as TensorFactory<T, V>
        }

        T::class == Int8::class && V::class == Byte::class -> {
            @Suppress("UNCHECKED_CAST")
            DefaultTensorFactories.getInt8Factory() as TensorFactory<T, V>
        }

        T::class == Int32::class && V::class == Int::class -> {
            @Suppress("UNCHECKED_CAST")
            DefaultTensorFactories.getInt32Factory() as TensorFactory<T, V>
        }

        else -> throw IllegalArgumentException(
            "Unsupported DType and value type combination: ${T::class.simpleName}, ${V::class.simpleName}. " +
                    "Supported combinations: (FP32, Float), (Int8, Byte), (Int32, Int)"
        )
    }

    return network(factory, content)
}

@NetworkDsl
public interface NetworkDslItem

/**
 * Core DSL interface for building neural networks with generic tensor types.
 * This interface provides a fluent API for constructing neural network architectures
 * with support for different data types and precision levels.
 *
 * @param T The data type (DType) that determines the precision and storage format
 * @param V The value type that corresponds to the native Kotlin type for the DType
 *
 * Type constraints ensure compatibility between DType and value type:
 * - T must extend DType to ensure valid tensor operations
 * - V should match the native type expected by the DType implementation
 *
 * Performance considerations:
 * - FP32/Float: Best accuracy, higher memory usage
 * - FP16/Float: Reduced memory, slightly lower accuracy
 * - Int8/Byte: Minimal memory, quantized operations
 * - Int32/Int: Integer operations, specific use cases
 */
@NetworkDsl
public interface NeuralNetworkDsl<T : DType, V> : NetworkDslItem {
    /**
     * Creates an input layer that defines the entry point for data into the network.
     *
     * @param inputSize The number of input features/dimensions
     * @param id Optional identifier for the layer (auto-generated if empty)
     */
    public fun input(inputSize: Int, id: String = "")

    /**
     * Creates a flatten layer that reshapes multi-dimensional tensors into 1D.
     * Useful for transitioning from convolutional to dense layers.
     *
     * @param id Optional identifier for the layer
     * @param content Configuration block for flatten-specific parameters
     */
    public fun flatten(id: String = "", content: FLATTEN<T, V>.() -> Unit = {})

    /**
     * Creates a dense (fully connected) layer with specified output dimension.
     *
     * @param outputDimension The number of neurons/output features
     * @param id Optional identifier for the layer
     * @param content Configuration block for weights, bias, and activation
     */
    public fun dense(outputDimension: Int, id: String = "", content: DENSE<T, V>.() -> Unit = {})

    /**
     * Creates a dense layer without specifying output dimension (must be set in content block).
     *
     * @param id Optional identifier for the layer
     * @param content Configuration block where output dimension, weights, and bias are set
     */
    public fun dense(id: String = "", content: DENSE<T, V>.() -> Unit = {})

    /**
     * Applies an activation function as a separate layer.
     *
     * @param id Optional identifier for the activation layer
     * @param activation Function that transforms tensor values (e.g., ReLU, Sigmoid)
     */
    public fun activation(id: String = "", activation: (Tensor<T, V>) -> Tensor<T, V>)

    /**
     * Groups layers into a sequential block for better organization.
     *
     * @param content DSL block containing the sequence of layers
     */
    public fun sequential(content: NeuralNetworkDsl<T, V>.() -> Unit)

    /**
     * Creates a named stage/block within the network for modular design.
     *
     * @param id Identifier for the stage
     * @param content DSL block containing the layers within this stage
     */
    public fun stage(id: String, content: NeuralNetworkDsl<T, V>.() -> Unit)
}

@NetworkDsl
public interface DENSE<T : DType, V> : NetworkDslItem {
    public var activation: (Tensor<T, V>) -> Tensor<T, V>
    public var units: Int
    public fun weights(initBlock: (Shape) -> Tensor<T, V>)
    public fun bias(initBlock: (Shape) -> Tensor<T, V>)

    // Factory-based convenience methods
    public val factory: TensorFactory<T, V>

    // Internal context for shape information
    public val weightsShape: Shape
    public val biasShape: Shape

    /**
     * Creates a tensor filled with zeros using the factory.
     */
    public fun zeros(shape: Shape): Tensor<T, V> = factory.zeros(shape)

    /**
     * Creates a tensor filled with ones using the factory.
     */
    public fun ones(shape: Shape): Tensor<T, V> = factory.ones(shape)

    /**
     * Creates a tensor filled with random values using the factory.
     */
    public fun random(shape: Shape): Tensor<T, V> = factory.random(shape)

    /**
     * Creates a tensor filled with random values using a specific seed.
     */
    public fun random(shape: Shape, seed: Long): Tensor<T, V> = factory.random(shape, seed)

    /**
     * Creates a tensor filled with random values using a specific Random instance.
     */
    public fun random(shape: Shape, random: kotlin.random.Random): Tensor<T, V> = factory.random(shape, random)

    /**
     * Creates a tensor filled with normally distributed random values.
     */
    public fun randomNormal(shape: Shape, mean: Double = 0.0, std: Double = 1.0): Tensor<T, V> =
        factory.randomNormal(shape, mean, std)

    /**
     * Creates a tensor filled with normally distributed random values using a specific seed.
     */
    public fun randomNormal(shape: Shape, mean: Double = 0.0, std: Double = 1.0, seed: Long): Tensor<T, V> =
        factory.randomNormal(shape, mean, std, seed)

    /**
     * Creates a tensor filled normally distributed random values using a specific Random instance.
     */
    public fun randomNormal(
        shape: Shape,
        mean: Double = 0.0,
        std: Double = 1.0,
        random: kotlin.random.Random
    ): Tensor<T, V> =
        factory.randomNormal(shape, mean, std, random)

    /**
     * Creates a tensor filled with uniformly distributed random values.
     */
    public fun randomUniform(shape: Shape, min: Double = 0.0, max: Double = 1.0): Tensor<T, V> =
        factory.randomUniform(shape, min, max)

    /**
     * Creates a tensor filled with uniformly distributed random values using a specific seed.
     */
    public fun randomUniform(shape: Shape, min: Double = 0.0, max: Double = 1.0, seed: Long): Tensor<T, V> =
        factory.randomUniform(shape, min, max, seed)

    /**
     * Creates a tensor filled with uniformly distributed random values using a specific Random instance.
     */
    public fun randomUniform(
        shape: Shape,
        min: Double = 0.0,
        max: Double = 1.0,
        random: kotlin.random.Random
    ): Tensor<T, V> =
        factory.randomUniform(shape, min, max, random)
}

// Extension functions for convenient parameterless initialization
/**
 * Extension function for weights initialization with implicit shape context.
 */
public fun <T : DType, V> DENSE<T, V>.weights(initBlock: WeightsScope<T, V>.() -> Tensor<T, V>) {
    val scope = WeightsScopeImpl(factory, weightsShape)
    weights { scope.initBlock() }
}

/**
 * Extension function for bias initialization with implicit shape context.
 */
public fun <T : DType, V> DENSE<T, V>.bias(initBlock: BiasScope<T, V>.() -> Tensor<T, V>) {
    val scope = BiasScopeImpl(factory, biasShape)
    bias { scope.initBlock() }
}

/**
 * Scope for weights initialization with implicit shape context.
 */
@NetworkDsl
public interface WeightsScope<T : DType, V> {
    public val factory: TensorFactory<T, V>
    public val shape: Shape

    public fun zeros(): Tensor<T, V> = factory.zeros(shape)
    public fun ones(): Tensor<T, V> = factory.ones(shape)
    public fun random(): Tensor<T, V> = factory.random(shape)
    public fun random(seed: Long): Tensor<T, V> = factory.random(shape, seed)
    public fun random(random: kotlin.random.Random): Tensor<T, V> = factory.random(shape, random)
    public fun randomNormal(mean: Double = 0.0, std: Double = 1.0): Tensor<T, V> =
        factory.randomNormal(shape, mean, std)

    public fun randomNormal(mean: Double = 0.0, std: Double = 1.0, seed: Long): Tensor<T, V> =
        factory.randomNormal(shape, mean, std, seed)

    public fun randomNormal(mean: Double = 0.0, std: Double = 1.0, random: kotlin.random.Random): Tensor<T, V> =
        factory.randomNormal(shape, mean, std, random)

    public fun randomUniform(min: Double = 0.0, max: Double = 1.0): Tensor<T, V> =
        factory.randomUniform(shape, min, max)

    public fun randomUniform(min: Double = 0.0, max: Double = 1.0, seed: Long): Tensor<T, V> =
        factory.randomUniform(shape, min, max, seed)

    public fun randomUniform(min: Double = 0.0, max: Double = 1.0, random: kotlin.random.Random): Tensor<T, V> =
        factory.randomUniform(shape, min, max, random)

    /**
     * Advanced initialization with custom random distribution.
     */
    public fun random(initBlock: (Shape) -> Tensor<T, V>): Tensor<T, V> = initBlock(shape)
}

/**
 * Scope for bias initialization with implicit shape context.
 */
@NetworkDsl
public interface BiasScope<T : DType, V> {
    public val factory: TensorFactory<T, V>
    public val shape: Shape

    public fun zeros(): Tensor<T, V> = factory.zeros(shape)
    public fun ones(): Tensor<T, V> = factory.ones(shape)
    public fun random(): Tensor<T, V> = factory.random(shape)
    public fun random(seed: Long): Tensor<T, V> = factory.random(shape, seed)
    public fun random(random: kotlin.random.Random): Tensor<T, V> = factory.random(shape, random)
    public fun randomNormal(mean: Double = 0.0, std: Double = 1.0): Tensor<T, V> =
        factory.randomNormal(shape, mean, std)

    public fun randomNormal(mean: Double = 0.0, std: Double = 1.0, seed: Long): Tensor<T, V> =
        factory.randomNormal(shape, mean, std, seed)

    public fun randomNormal(mean: Double = 0.0, std: Double = 1.0, random: kotlin.random.Random): Tensor<T, V> =
        factory.randomNormal(shape, mean, std, random)

    public fun randomUniform(min: Double = 0.0, max: Double = 1.0): Tensor<T, V> =
        factory.randomUniform(shape, min, max)

    public fun randomUniform(min: Double = 0.0, max: Double = 1.0, seed: Long): Tensor<T, V> =
        factory.randomUniform(shape, min, max, seed)

    public fun randomUniform(min: Double = 0.0, max: Double = 1.0, random: kotlin.random.Random): Tensor<T, V> =
        factory.randomUniform(shape, min, max, random)

    /**
     * Advanced initialization with custom random distribution.
     */
    public fun random(initBlock: (Shape) -> Tensor<T, V>): Tensor<T, V> = initBlock(shape)
}

/**
 * Implementation of WeightsScope for weights initialization.
 */
public class WeightsScopeImpl<T : DType, V>(
    override val factory: TensorFactory<T, V>,
    override val shape: Shape
) : WeightsScope<T, V>

/**
 * Implementation of BiasScope for bias initialization.
 */
public class BiasScopeImpl<T : DType, V>(
    override val factory: TensorFactory<T, V>,
    override val shape: Shape
) : BiasScope<T, V>

@NetworkDsl
public interface FLATTEN<T : DType, V> : NetworkDslItem {
    public var startDim: Int
    public var endDim: Int
}

private fun getDefaultName(id: String, s: String, size: Int): String {
    if (id.isNotEmpty()) return id
    return "$s-$size"
}

public class FlattenImpl<T : DType, V>(
    override var startDim: Int = 1,
    override var endDim: Int = -1,
    private val id: String
) : FLATTEN<T, V> {
    public fun create(): Module<T, V> {
        return Flatten(startDim, endDim, id)
    }
}

private fun <T : DType, V> createLinear(
    inFeatures: Int,
    outFeatures: Int,
    id: String,
    myInitWeights: Tensor<T, V>? = null,
    myInitBias: Tensor<T, V>? = null,
    factory: TensorFactory<T, V>
): Linear<T, V> {
    return when {
        myInitWeights != null && myInitBias != null ->
            Linear(
                inFeatures = inFeatures,
                outFeatures = outFeatures,
                name = id,
                initWeights = myInitWeights,
                initBias = myInitBias
            )

        myInitWeights == null && myInitBias != null -> {

            val safeWeights = factory.zeros(Shape(outFeatures, inFeatures))

            Linear(
                inFeatures = inFeatures,
                outFeatures = outFeatures,
                name = id,
                initWeights = safeWeights,
                initBias = myInitBias
            )
        }

        myInitWeights != null && myInitBias == null -> {
            val safeBias = factory.zeros(Shape(outFeatures))

            Linear(
                inFeatures = inFeatures,
                outFeatures = outFeatures,
                name = id,
                initWeights = myInitWeights,
                initBias = safeBias
            )
        }

        else -> {
            val safeWeights = factory.zeros(Shape(outFeatures, inFeatures))
            val safeBias = factory.zeros(Shape(outFeatures))

            Linear(
                inFeatures = inFeatures,
                outFeatures = outFeatures,
                name = id,
                initWeights = safeWeights,
                initBias = safeBias
            )
        }
    }
}


public class DenseImpl<T : DType, V>(
    private val inputDimension: Int,
    private var _outputDimension: Int,
    private val id: String,
    override val factory: TensorFactory<T, V>
) : DENSE<T, V> {

    private var weightsValue: Tensor<T, V>? = null
    private var biasValue: Tensor<T, V>? = null
    private var _activation: (Tensor<T, V>) -> Tensor<T, V> = { tensor -> tensor }

    // Expose the output dimension
    public val outputDimension: Int
        get() = _outputDimension

    // Shape context for the DSL
    override val weightsShape: Shape
        get() = Shape(_outputDimension, inputDimension)

    override val biasShape: Shape
        get() = Shape(_outputDimension)

    public fun create(): List<Module<T, V>> {
        // Create default tensors if not provided - use factory for defaults
        val weights = weightsValue
        val bias = biasValue

        val linear = createLinear(
            inFeatures = inputDimension,
            outFeatures = _outputDimension,
            id = id,
            myInitWeights = weights,
            myInitBias = bias,
            factory = factory
        )

        return listOf(
            linear,
            ActivationsWrapperModule(activation, "activation")
        )
    }

    override var activation: (Tensor<T, V>) -> Tensor<T, V>
        get() = _activation
        set(value) {
            _activation = value
        }

    override var units: Int
        get() = _outputDimension
        set(value) {
            _outputDimension = value
        }

    override fun weights(initBlock: (Shape) -> Tensor<T, V>) {
        weightsValue = initBlock(weightsShape)
    }

    override fun bias(initBlock: (Shape) -> Tensor<T, V>) {
        biasValue = initBlock(biasShape)
    }

}

// Stage implementation
public class StageImpl<T : DType, V>(
    private val id: String,
    private val factory: TensorFactory<T, V>
) : NeuralNetworkDsl<T, V> {
    public val modules: MutableList<Module<T, V>> = mutableListOf<Module<T, V>>()
    public var lastDimension: Int = 0
    public var inputDimension: Int = 0

    public fun create(): Module<T, V> = MLP(*modules.toTypedArray(), name = id)

    override fun input(inputSize: Int, id: String) {
        lastDimension = inputSize
        modules.add(Input(Shape(inputSize), name = getDefaultName(id, "Input", modules.size)))
    }

    override fun flatten(id: String, content: FLATTEN<T, V>.() -> Unit) {
        val impl = FlattenImpl<T, V>(
            id = getDefaultName(id, "flatten", modules.size)
        )
        impl.content()
        modules += impl.create()
    }

    override fun dense(outputDimension: Int, id: String, content: DENSE<T, V>.() -> Unit) {
        val inputDimension = lastDimension
        lastDimension = outputDimension
        val impl = DenseImpl(
            inputDimension = inputDimension,
            _outputDimension = outputDimension,
            id = getDefaultName(id, "linear", modules.size),
            factory = factory
        )
        impl.content()
        // dense layer consists of linear module and activation function module (2 modules)
        modules += impl.create()
    }

    override fun dense(id: String, content: DENSE<T, V>.() -> Unit) {
        // This version of dense requires units to be specified in the content block
        val impl = DenseImpl(
            inputDimension = lastDimension,
            _outputDimension = 0, // Will be set in content block via units property
            id = getDefaultName(id, "linear", modules.size),
            factory = factory
        )
        impl.content()
        // Update lastDimension based on the units set in the content block
        lastDimension = impl.outputDimension
        // dense layer consists of linear module and activation function module (2 modules)
        modules += impl.create()
    }

    override fun activation(id: String, activation: (Tensor<T, V>) -> Tensor<T, V>) {
        modules += ActivationsWrapperModule(activation, getDefaultName(id, "activation", modules.size))
    }

    override fun sequential(content: NeuralNetworkDsl<T, V>.() -> Unit) {
        val sequentialImpl = NeuralNetworkDslImpl(factory)
        sequentialImpl.lastDimension = lastDimension
        sequentialImpl.content()
        lastDimension = sequentialImpl.lastDimension
        modules += sequentialImpl.create()
    }

    override fun stage(id: String, content: NeuralNetworkDsl<T, V>.() -> Unit) {
        val stageImpl = StageImpl(id, factory)
        stageImpl.lastDimension = lastDimension
        stageImpl.content()
        lastDimension = stageImpl.lastDimension
        modules += stageImpl.create()
    }
}

private class NeuralNetworkDslImpl<T : DType, V>(
    private val factory: TensorFactory<T, V>
) : NeuralNetworkDsl<T, V> {

    val modules = mutableListOf<Module<T, V>>()
    var lastDimension = 0

    fun create(): Module<T, V> = NetworkBuilder<T, V>().add(*modules.toTypedArray()).build()

    override fun input(inputSize: Int, id: String) {
        lastDimension = inputSize
        modules.add(Input(Shape(inputSize), name = getDefaultName(id, "Input", modules.size)))
    }

    override fun flatten(id: String, content: FLATTEN<T, V>.() -> Unit) {
        val impl = FlattenImpl<T, V>(
            id = getDefaultName(id, "flatten", modules.size)
        )
        impl.content()
        modules += impl.create()
    }

    override fun dense(outputDimension: Int, id: String, content: DENSE<T, V>.() -> Unit) {
        val inputDimension = lastDimension
        lastDimension = outputDimension
        val impl = DenseImpl<T, V>(
            inputDimension = inputDimension,
            _outputDimension = outputDimension,
            id = getDefaultName(id, "linear", modules.size),
            factory = factory
        )
        impl.content()
        // dense layer consists of linear module and activation function module (2 modules)
        modules += impl.create()
    }

    override fun dense(id: String, content: DENSE<T, V>.() -> Unit) {
        // This version of dense requires units to be specified in the content block
        val impl = DenseImpl<T, V>(
            inputDimension = lastDimension,
            _outputDimension = 0, // Will be set in content block via units property
            id = getDefaultName(id, "linear", modules.size),
            factory = factory
        )
        impl.content()
        // Update lastDimension based on the units set in the content block
        lastDimension = impl.outputDimension
        // dense layer consists of linear module and activation function module (2 modules)
        modules += impl.create()
    }

    override fun activation(id: String, activation: (Tensor<T, V>) -> Tensor<T, V>) {
        modules += ActivationsWrapperModule(activation, getDefaultName(id, "activation", modules.size))
    }

    override fun sequential(content: NeuralNetworkDsl<T, V>.() -> Unit) {
        val sequentialImpl = NeuralNetworkDslImpl(factory)
        sequentialImpl.lastDimension = lastDimension
        sequentialImpl.content()
        lastDimension = sequentialImpl.lastDimension
        modules += sequentialImpl.create()
    }

    override fun stage(id: String, content: NeuralNetworkDsl<T, V>.() -> Unit) {
        val stageImpl = StageImpl(id, factory)
        stageImpl.lastDimension = lastDimension
        stageImpl.content()
        lastDimension = stageImpl.lastDimension
        modules += stageImpl.create()
    }
}


@NetworkDsl
public class NetworkBuilder<T : DType, V> {
    private val modules = mutableListOf<Module<T, V>>()

    public fun add(vararg modules: Module<T, V>): NetworkBuilder<T, V> {
        this.modules += modules.toList()
        return this
    }

    public fun build(): Module<T, V> = MLP(*modules.toTypedArray(), name = "MLP")
}