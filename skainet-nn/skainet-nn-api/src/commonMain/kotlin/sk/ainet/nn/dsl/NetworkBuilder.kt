package sk.ainet.nn.dsl

import sk.ainet.nn.activations.ActivationsWrapperModule
import sk.ainet.core.tensor.Shape
import sk.ainet.core.tensor.Tensor
import sk.ainet.core.tensor.DType
import sk.ainet.core.tensor.FP32
import sk.ainet.nn.Flatten
import sk.ainet.nn.Input
import sk.ainet.nn.Linear
import sk.ainet.nn.Module
import sk.ainet.nn.topology.MLP


// DSL Marker to restrict the DSL to its intended scope
@DslMarker
public annotation class NetworkDsl

// Generic network builder function
@NetworkDsl
public fun <T : DType, V> network(content: NeuralNetworkDsl<T, V>.() -> Unit): Module<T, V> = NeuralNetworkDslImpl<T, V>()
    .apply(content)
    .create()

// Backward compatibility - keep existing non-generic function for FP32/Float
@NetworkDsl
public fun network(content: NeuralNetworkDsl<FP32, Float>.() -> Unit): Module<FP32, Float> = network<FP32, Float>(content)

// Helper functions for common types
@NetworkDsl
public fun networkFP32(content: NeuralNetworkDsl<FP32, Float>.() -> Unit): Module<FP32, Float> = network<FP32, Float>(content)

@NetworkDsl
public interface NetworkDslItem

@NetworkDsl
public interface NeuralNetworkDsl<T : DType, V> : NetworkDslItem {
    public fun input(inputSize: Int, id: String = "")

    public fun flatten(id: String = "", content: FLATTEN<T, V>.() -> Unit = {})

    public fun dense(outputDimension: Int, id: String = "", content: DENSE<T, V>.() -> Unit = {})

    public fun dense(id: String = "", content: DENSE<T, V>.() -> Unit = {})

    public fun activation(id: String = "", activation: (Tensor<T, V>) -> Tensor<T, V>)

    public fun sequential(content: NeuralNetworkDsl<T, V>.() -> Unit)

    public fun stage(id: String, content: NeuralNetworkDsl<T, V>.() -> Unit)
}

@NetworkDsl
public interface DENSE<T : DType, V> : NetworkDslItem {
    public var activation: (Tensor<T, V>) -> Tensor<T, V>
    public var units: Int
    public fun weights(initBlock: (Shape) -> Tensor<T, V>)
    public fun bias(initBlock: (Shape) -> Tensor<T, V>)
}

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

public class DenseImpl<T : DType, V>(
    private val inputDimension: Int, 
    private var _outputDimension: Int, 
    private val id: String
) : DENSE<T, V> {

    private var weightsValue: Tensor<T, V>? = null
    private var biasValue: Tensor<T, V>? = null
    private var _activation: (Tensor<T, V>) -> Tensor<T, V> = { tensor -> tensor }

    // Expose the output dimension
    public val outputDimension: Int
        get() = _outputDimension

    public fun create(): List<Module<T, V>> {
        // Create default tensors if not provided
        val weights = weightsValue ?: throw IllegalStateException("Weights must be initialized")
        val bias = biasValue ?: throw IllegalStateException("Bias must be initialized")
        
        val linear = Linear<T, V>(
            inFeatures = inputDimension,
            outFeatures = _outputDimension,
            name = id,
            initWeights = weights,
            initBias = bias
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
        weightsValue = initBlock(Shape(_outputDimension, inputDimension))
    }

    override fun bias(initBlock: (Shape) -> Tensor<T, V>) {
        biasValue = initBlock(Shape(_outputDimension))
    }
}

// Stage implementation
public class StageImpl<T : DType, V>(private val id: String) : NeuralNetworkDsl<T, V> {
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
        val impl = DenseImpl<T, V>(
            inputDimension = inputDimension,
            _outputDimension = outputDimension,
            id = getDefaultName(id, "linear", modules.size)
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
            id = getDefaultName(id, "linear", modules.size)
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
        val sequentialImpl = NeuralNetworkDslImpl<T, V>()
        sequentialImpl.lastDimension = lastDimension
        sequentialImpl.content()
        lastDimension = sequentialImpl.lastDimension
        modules += sequentialImpl.create()
    }

    override fun stage(id: String, content: NeuralNetworkDsl<T, V>.() -> Unit) {
        val stageImpl = StageImpl<T, V>(id)
        stageImpl.lastDimension = lastDimension
        stageImpl.content()
        lastDimension = stageImpl.lastDimension
        modules += stageImpl.create()
    }
}

private class NeuralNetworkDslImpl<T : DType, V> : NeuralNetworkDsl<T, V> {

    public val modules = mutableListOf<Module<T, V>>()
    public var lastDimension = 0

    public fun create(): Module<T, V> = NetworkBuilder<T, V>().add(*modules.toTypedArray()).build()

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
            id = getDefaultName(id, "linear", modules.size)
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
            id = getDefaultName(id, "linear", modules.size)
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
        val sequentialImpl = NeuralNetworkDslImpl<T, V>()
        sequentialImpl.lastDimension = lastDimension
        sequentialImpl.content()
        lastDimension = sequentialImpl.lastDimension
        modules += sequentialImpl.create()
    }

    override fun stage(id: String, content: NeuralNetworkDsl<T, V>.() -> Unit) {
        val stageImpl = StageImpl<T, V>(id)
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

    public fun build(): Module<T, V> = MLP(*modules.toTypedArray(), name = "Sequential")
}