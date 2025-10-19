package sk.ainet.lang.nn.dsl

import sk.ainet.lang.nn.activations.ActivationsWrapperModule
import sk.ainet.lang.nn.Flatten
import sk.ainet.lang.nn.Input
import sk.ainet.lang.nn.Linear
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.topology.MLP
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.types.DType
import kotlin.random.Random
import kotlin.reflect.KClass


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
public inline fun <reified T : DType, V> network(
    factory: TensorDataFactory,
    content: NeuralNetworkDsl<T, V>.() -> Unit
): Module<T, V> =
    NeuralNetworkDslImpl<T, V>(T::class, factory)
        .apply(content)
        .create()

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
     * Creates a dense layer with precision override and specified output dimension.
     * This allows individual layers to use different precision than the network default.
     *
     * @param TLayer The precision type for this specific layer
     * @param outputDimension The number of neurons/output features
     * @param id Optional identifier for the layer
     * @param content Configuration block for weights, bias, and activation
     */
    public fun <TLayer : DType> dense(
        outputDimension: Int,
        id: String = "",
        content: DENSE<TLayer, V>.() -> Unit = {}
    ): Module<T, V>

    /**
     * Creates a dense layer with precision override without specifying output dimension.
     *
     * @param TLayer The precision type for this specific layer
     * @param id Optional identifier for the layer
     * @param content Configuration block where output dimension, weights, and bias are set
     */
    public fun <TLayer : DType> dense(
        id: String = "",
        content: DENSE<TLayer, V>.() -> Unit = {}
    ): Module<T, V>

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

    /**
     * Creates a convolutional block with specific precision type.
     * This allows for mixed-precision networks where convolutional layers
     * can use different precision than the network default.
     *
     * @param TBlock The precision type for this convolutional block
     * @param content DSL block containing convolutional layers with TBlock precision
     * @return Module that handles the precision conversion automatically
     */
    /*
    public fun <TBlock : DType> convBlock(
        content: ConvBlockDsl<TBlock, V>.() -> Unit
    ): Module<T, V>

     */

    /**
     * Creates a transformer block with specific precision type.
     * This allows for mixed-precision networks where transformer layers
     * can use different precision than the network default.
     *
     * @param TBlock The precision type for this transformer block
     * @param content DSL block containing transformer layers with TBlock precision
     * @return Module that handles the precision conversion automatically
     */
    /*
    public fun <TBlock : DType> transformerBlock(
        content: TransformerBlockDsl<TBlock, V>.() -> Unit
    ): Module<T, V>

     */

    /**
     * Creates a precision-scoped stage within the network.
     * This allows grouping layers with a specific precision type that differs
     * from the network default, enabling fine-grained mixed-precision control.
     *
     * Unlike the basic stage method, this version allows changing the precision
     * type for all layers within the stage scope.
     *
     * @param TStage The precision type for all layers within this stage
     * @param id Identifier for the stage
     * @param content DSL block containing layers with TStage precision
     * @return Module that handles the precision conversion automatically
     */
    public fun <TStage : DType> stage(
        id: String,
        content: NeuralNetworkDsl<TStage, V>.() -> Unit
    ): Module<T, V>
}

@NetworkDsl
public interface DENSE<T : DType, V> : NetworkDslItem {
    public var activation: (Tensor<T, V>) -> Tensor<T, V>
    public var units: Int

    //public fun weights(initBlock: (Shape) -> Tensor<T, V>)
    public fun weights(initBlock: WeightsScope<T, V>.(Shape) -> Tensor<T, V>)
    public fun bias(initBlock: BiasScope<T, V>.(Shape) -> Tensor<T, V>)

    // Factory-based convenience methods
    public val factory: TensorDataFactory

    // Internal context for shape information
    public val weightsShape: Shape
    public val biasShape: Shape


}

/**
 * Base scope providing common tensor creation and initialization methods.
 */
@NetworkDsl
public interface TensorsValueScope<T : DType, V> {
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
    public fun randn(mean: Float = 0.0f, std: Float = 1.0f, random: Random = Random.Default): Tensor<T, V> {
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
        val data = factory.init<T, V>(shape, dtype, generator)
        return VoidOpsTensor(data, dtype)
    }
    
    /**
     * Create tensor with custom random initialization
     */
    public fun randomInit(generator: (random: Random) -> V, random: Random = Random.Default): Tensor<T, V> {
        val data = factory.randomInit<T, V>(shape, dtype, generator, random)
        return VoidOpsTensor(data, dtype)
    }

    /**
     * Factories
     */
    // fromXX Float
    public fun from(vararg data: Float): Tensor<T, V> = fromArray(data.toTypedArray().toFloatArray())
    public fun fromList(data: List<Float>): Tensor<T, V> = fromArray(data.toFloatArray())
    public fun fromArray(data: FloatArray): Tensor<T, V> {
        require(data.size == shape.volume) {
            "Data size ${data.size} doesn't match shape volume ${shape.volume}"
        }
        val tensorData = factory.fromFloatArray<T, V>(shape, dtype, data)
        return VoidOpsTensor(tensorData, dtype)
    }

    // fromXX Int
    public fun from(vararg data: Int): Tensor<T, V> = fromArray(data.toTypedArray().toIntArray())
    public fun fromIntList(data: List<Int>): Tensor<T, V> = fromArray(data.toIntArray())
    public fun fromArray(data: IntArray): Tensor<T, V> {
        require(data.size == shape.volume) {
            "Data size ${data.size} doesn't match shape volume ${shape.volume}"
        }
        val tensorData = factory.fromIntArray<T, V>(shape, dtype, data)
        return VoidOpsTensor(tensorData, dtype)
    }


    /**
     * Advanced initialization with custom random distribution.
     */
    public fun random(initBlock: (Shape) -> Tensor<T, V>): Tensor<T, V> = initBlock(shape)

}

/**
 * Scope for weights initialization with implicit shape context.
 */
@NetworkDsl
public interface WeightsScope<T : DType, V> : TensorsValueScope<T, V>

/**
 * Scope for bias initialization with implicit shape context.
 */
@NetworkDsl
public interface BiasScope<T : DType, V> : TensorsValueScope<T, V>

/**
 * Implementation of WeightsScope for weights initialization.
 */
public class WeightsScopeImpl<T : DType, V>(
    override val factory: TensorDataFactory,
    override val shape: Shape,
    override val dtype: KClass<T>
) : WeightsScope<T, V>

/**
 * Implementation of BiasScope for bias initialization.
 */
public class BiasScopeImpl<T : DType, V>(
    override val factory: TensorDataFactory,
    override val shape: Shape,
    override val dtype: KClass<T>
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
    kClass: kotlin.reflect.KClass<T>,
    myInitWeights: Tensor<T, V>? = null,
    myInitBias: Tensor<T, V>? = null,
    factory: TensorDataFactory
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

            val safeWeights = factory.zeros<T, V>(Shape(outFeatures, inFeatures), kClass)

            Linear(
                inFeatures = inFeatures,
                outFeatures = outFeatures,
                name = id,
                initWeights = VoidOpsTensor(safeWeights, kClass),
                initBias = myInitBias
            )
        }

        myInitWeights != null && myInitBias == null -> {
            val safeBias = factory.zeros<T, V>(Shape(outFeatures), kClass)

            Linear(
                inFeatures = inFeatures,
                outFeatures = outFeatures,
                name = id,
                initWeights = myInitWeights,
                initBias = VoidOpsTensor(safeBias, kClass)
            )
        }

        else -> {
            val safeWeights = factory.zeros<T, V>(Shape(outFeatures, inFeatures), kClass)
            val safeBias = factory.zeros<T, V>(Shape(outFeatures), kClass)

            Linear(
                inFeatures = inFeatures,
                outFeatures = outFeatures,
                name = id,
                initWeights = VoidOpsTensor(safeWeights, kClass),
                initBias = VoidOpsTensor(safeBias, kClass)
            )
        }
    }
}


public class DenseImpl<T : DType, V>(
    private val inputDimension: Int,
    private var _outputDimension: Int,
    private val id: String,
    private val kClass: kotlin.reflect.KClass<T>,
    override val factory: TensorDataFactory
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
            kClass = kClass,
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

    private fun initWeights(tensor: Tensor<T, V>) {
        tensor

    }

    override fun weights(initBlock: WeightsScope<T, V>.(Shape) -> Tensor<T, V>) {
        val scope = WeightsScopeImpl<T, V>(factory, weightsShape, kClass)
        weightsValue = scope.initBlock(weightsShape)
    }

    override fun bias(initBlock: BiasScope<T, V>.(Shape) -> Tensor<T, V>) {
        val scope = BiasScopeImpl<T, V>(factory, biasShape, kClass)
        biasValue = scope.initBlock(biasShape)
    }
}



// Stage implementation
public class StageImpl<T : DType, V>(
    private val id: String,
    private val kClass: kotlin.reflect.KClass<T>,
    private val factory: TensorDataFactory
) : NeuralNetworkDsl<T, V> {
    public val modules: MutableList<Module<T, V>> = mutableListOf<Module<T, V>>()
    public var lastDimension: Int = 0
    public var inputDimension: Int = 0

    public fun create(): Module<T, V> = MLP(*modules.toTypedArray(), name = id)

    override fun input(inputSize: Int, id: String) {
        lastDimension = inputSize
        modules.add(Input(name = getDefaultName(id, "Input", modules.size)))
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
            kClass = kClass,
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
            kClass = kClass,
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
        val sequentialImpl = NeuralNetworkDslImpl<T, V>(kClass, factory)
        sequentialImpl.lastDimension = lastDimension
        sequentialImpl.content()
        lastDimension = sequentialImpl.lastDimension
        modules += sequentialImpl.create()
    }

    override fun stage(id: String, content: NeuralNetworkDsl<T, V>.() -> Unit) {
        val stageImpl = StageImpl<T, V>(id, kClass, factory)
        stageImpl.lastDimension = lastDimension
        stageImpl.content()
        lastDimension = stageImpl.lastDimension
        modules += stageImpl.create()
    }

    override fun <TLayer : DType> dense(
        outputDimension: Int,
        id: String,
        content: DENSE<TLayer, V>.() -> Unit
    ): Module<T, V> {
        // Create a mixed-precision module that handles conversion
        TODO("Mixed-precision dense implementation needed")
    }

    override fun <TLayer : DType> dense(
        id: String,
        content: DENSE<TLayer, V>.() -> Unit
    ): Module<T, V> {
        // Create a mixed-precision module that handles conversion
        TODO("Mixed-precision dense implementation needed")
    }

    override fun <TStage : DType> stage(
        id: String,
        content: NeuralNetworkDsl<TStage, V>.() -> Unit
    ): Module<T, V> {
        // Create a mixed-precision stage that handles conversion
        TODO("Mixed-precision stage implementation needed")
    }

    /*
    override fun <TBlock : DType> convBlock(
        content: ConvBlockDsl<TBlock, V>.() -> Unit
    ): Module<T, V> {
        // Create a mixed-precision convolutional block
        TODO("Mixed-precision convBlock implementation needed")
    }

     */


    /*
    override fun <TBlock : DType> transformerBlock(
        content: TransformerBlockDsl<TBlock, V>.() -> Unit
    ): Module<T, V> {
        // Create a mixed-precision transformer block
        TODO("Mixed-precision transformerBlock implementation needed")
    }

     */
}

public class NeuralNetworkDslImpl<T : DType, V>(
    private val kClass: kotlin.reflect.KClass<T>,
    private val factory: TensorDataFactory
) : NeuralNetworkDsl<T, V> {

    public val modules: MutableList<Module<T, V>> = mutableListOf<Module<T, V>>()
    public var lastDimension: Int = 0

    public fun create(): Module<T, V> = NetworkBuilder<T, V>().add(*modules.toTypedArray()).build()

    override fun input(inputSize: Int, id: String) {
        lastDimension = inputSize
        modules.add(Input(name = getDefaultName(id, "Input", modules.size)))
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
            kClass = kClass,
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
            kClass = kClass,
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
        val sequentialImpl = NeuralNetworkDslImpl<T, V>(kClass, factory)
        sequentialImpl.lastDimension = lastDimension
        sequentialImpl.content()
        lastDimension = sequentialImpl.lastDimension
        modules += sequentialImpl.create()
    }

    override fun stage(id: String, content: NeuralNetworkDsl<T, V>.() -> Unit) {
        val stageImpl = StageImpl<T, V>(id, kClass, factory)
        stageImpl.lastDimension = lastDimension
        stageImpl.content()
        lastDimension = stageImpl.lastDimension
        modules += stageImpl.create()
    }

    override fun <TLayer : DType> dense(
        outputDimension: Int,
        id: String,
        content: DENSE<TLayer, V>.() -> Unit
    ): Module<T, V> {
        // Create a mixed-precision module that handles conversion
        TODO("Mixed-precision dense implementation needed")
    }

    override fun <TLayer : DType> dense(
        id: String,
        content: DENSE<TLayer, V>.() -> Unit
    ): Module<T, V> {
        // Create a mixed-precision module that handles conversion
        TODO("Mixed-precision dense implementation needed")
    }

    override fun <TStage : DType> stage(
        id: String,
        content: NeuralNetworkDsl<TStage, V>.() -> Unit
    ): Module<T, V> {
        // Create a mixed-precision stage that handles conversion
        TODO("Mixed-precision stage implementation needed")
    }

    /*
    override fun <TBlock : DType> convBlock(
        content: ConvBlockDsl<TBlock, V>.() -> Unit
    ): Module<T, V> {
        // Create a mixed-precision convolutional block
        TODO("Mixed-precision convBlock implementation needed")
    }

     */

    /*
    override fun <TBlock : DType> transformerBlock(
        content: TransformerBlockDsl<TBlock, V>.() -> Unit
    ): Module<T, V> {
        // Create a mixed-precision transformer block
        TODO("Mixed-precision transformerBlock implementation needed")
    }

     */
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
