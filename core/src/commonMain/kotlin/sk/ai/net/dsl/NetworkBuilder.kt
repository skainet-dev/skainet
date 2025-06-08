package sk.ai.net.dsl

import sk.ai.net.nn.activations.ActivationsWrapperModule
import sk.ai.net.Shape
import sk.ai.net.Tensor
import sk.ai.net.nn.Flatten
import sk.ai.net.nn.Input
import sk.ai.net.nn.Linear
import sk.ai.net.nn.Conv2d
import sk.ai.net.nn.MaxPool2d
import sk.ai.net.nn.Module
import sk.ai.net.nn.topology.MLP


// DSL Marker to restrict the DSL to its intended scope
@DslMarker
annotation class NetworkDsl

@NetworkDsl
fun network(content: NeuralNetworkDsl.() -> Unit) = NeuralNetworkDslImpl()
    .apply(content)
    .create()

@NetworkDsl
interface NetworkDslItem

@NetworkDsl
interface NeuralNetworkDsl : NetworkDslItem {
    fun input(inputSize: Int, id: String = "")

    fun flatten(id: String = "", content: FLATTEN.() -> Unit = {})

    fun conv2d(id: String = "", content: CONV2D.() -> Unit = {})

    fun maxPool2d(id: String = "", content: MAXPOOL2D.() -> Unit = {})

    fun dense(outputDimension: Int, id: String = "", content: DENSE.() -> Unit = {})
    
    fun dense(id: String = "", content: DENSE.() -> Unit = {})

    fun activation(id: String = "", activation: (Tensor) -> Tensor)
    
    fun sequential(content: NeuralNetworkDsl.() -> Unit)
    
    fun stage(id: String, content: NeuralNetworkDsl.() -> Unit)
}

@NetworkDsl
interface DENSE : NetworkDslItem {
    var activation: (Tensor) -> Tensor
    var units: Int
    fun weights(initBlock: (Shape) -> Tensor)
    fun bias(initBlock: (Shape) -> Tensor)
}

@NetworkDsl
interface FLATTEN : NetworkDslItem {
    var startDim: Int
    var endDim: Int
}

@NetworkDsl
interface CONV2D : NetworkDslItem {
    var outChannels: Int
    var kernelSize: Int
    var stride: Int
    var padding: Int
}

@NetworkDsl
interface MAXPOOL2D : NetworkDslItem {
    var kernelSize: Int
    var stride: Int
}


private fun getDefaultName(id: String, s: String, size: Int): String {
    if (id.isNotEmpty()) return id
    return "$s-$size"
}


fun createLinear(
    inFeatures: Int,
    outFeatures: Int,
    id: String,
    myInitWeights: Tensor? = null,
    myInitBias: Tensor? = null
): Linear {
    return when {
        myInitWeights != null && myInitBias != null ->
            Linear(
                inFeatures = inFeatures,
                outFeatures = outFeatures,
                name = id,
                initWeights = myInitWeights,
                initBias = myInitBias
            )

        myInitWeights != null ->
            Linear(inFeatures = inFeatures, outFeatures = outFeatures, name = id, initWeights = myInitWeights)

        myInitBias != null ->
            Linear(inFeatures = inFeatures, outFeatures = outFeatures, name = id, initBias = myInitBias)

        else ->
            Linear(inFeatures = inFeatures, outFeatures = outFeatures, name = id)
    }
}

class FlattenImpl(
    override var startDim: Int = 1,
    override var endDim: Int = -1,
    private val id: String
) : FLATTEN {
    fun create(): Module {
        return Flatten(startDim, endDim, id)
    }
}

class DenseImpl(
    private val inputDimension: Int, private var _outputDimension: Int, private val id: String
) : DENSE {

    private var weightsValue: Tensor? = null
    private var biasValue: Tensor? = null
    private var _activation: (Tensor) -> Tensor = { tensor -> tensor }

    // Expose the output dimension
    val outputDimension: Int
        get() = _outputDimension

    fun create(): List<Module> {
        val linear = createLinear(
            inFeatures = inputDimension,
            outFeatures = _outputDimension,
            id = id,
            myInitWeights = weightsValue,
            myInitBias = biasValue
        )

        return listOf(
            linear,
            ActivationsWrapperModule(activation, "activation")
        )
    }

    override var activation: (Tensor) -> Tensor
        get() = _activation
        set(value) {
            _activation = value
        }

    override var units: Int
        get() = _outputDimension
        set(value) {
            _outputDimension = value
        }

    override fun weights(initBlock: (Shape) -> Tensor) {
        weightsValue = initBlock(Shape(_outputDimension, inputDimension))
    }

    override fun bias(initBlock: (Shape) -> Tensor) {
        biasValue = initBlock(Shape(_outputDimension))
    }
}

class Conv2dImpl(
    private val inChannels: Int,
    override var outChannels: Int = 1,
    override var kernelSize: Int = 3,
    override var stride: Int = 1,
    override var padding: Int = 0,
    private val id: String
) : CONV2D {
    fun create(): Module = Conv2d(
        inChannels = inChannels,
        outChannels = outChannels,
        kernelSize = kernelSize,
        stride = stride,
        padding = padding,
        name = id
    )
}

class MaxPool2dImpl(
    override var kernelSize: Int = 2,
    override var stride: Int = 2,
    private val id: String
) : MAXPOOL2D {
    fun create(): Module = MaxPool2d(
        kernelSize = kernelSize,
        stride = stride,
        name = id
    )
}

// Stage implementation
class StageImpl(private val id: String) : NeuralNetworkDsl {
    val modules = mutableListOf<Module>()
    var lastDimension = 0
    var inputDimension = 0

    fun create(): Module = MLP(*modules.toTypedArray(), name = id)

    override fun input(inputSize: Int, id: String) {
        lastDimension = inputSize
        modules.add(Input(Shape(inputSize), name = getDefaultName(id, "Input", modules.size)))
    }

    override fun flatten(id: String, content: FLATTEN.() -> Unit) {
        val impl = FlattenImpl(
            id = getDefaultName(id, "flatten", modules.size)
        )
        impl.content()
        modules += impl.create()
    }

    override fun conv2d(id: String, content: CONV2D.() -> Unit) {
        val impl = Conv2dImpl(
            inChannels = lastDimension,
            id = getDefaultName(id, "conv2d", modules.size)
        )
        impl.content()
        lastDimension = impl.outChannels
        modules += impl.create()
    }

    override fun maxPool2d(id: String, content: MAXPOOL2D.() -> Unit) {
        val impl = MaxPool2dImpl(
            id = getDefaultName(id, "maxPool2d", modules.size)
        )
        impl.content()
        modules += impl.create()
    }

    override fun dense(outputDimension: Int, id: String, content: DENSE.() -> Unit) {
        val inputDimension = lastDimension
        lastDimension = outputDimension
        val impl = DenseImpl(
            inputDimension = inputDimension,
            _outputDimension = outputDimension,
            id = getDefaultName(id, "linear", modules.size)
        )
        impl.content()
        // dense layer consists of linear module and activation function module (2 modules)
        modules += impl.create()
    }

    override fun dense(id: String, content: DENSE.() -> Unit) {
        // This version of dense requires units to be specified in the content block
        val impl = DenseImpl(
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

    override fun activation(id: String, activation: (Tensor) -> Tensor) {
        modules += ActivationsWrapperModule(activation, getDefaultName(id, "activation", modules.size))
    }

    override fun sequential(content: NeuralNetworkDsl.() -> Unit) {
        val sequentialImpl = NeuralNetworkDslImpl()
        sequentialImpl.lastDimension = lastDimension
        sequentialImpl.content()
        lastDimension = sequentialImpl.lastDimension
        modules += sequentialImpl.create()
    }

    override fun stage(id: String, content: NeuralNetworkDsl.() -> Unit) {
        val stageImpl = StageImpl(id)
        stageImpl.lastDimension = lastDimension
        stageImpl.content()
        lastDimension = stageImpl.lastDimension
        modules += stageImpl.create()
    }
}

private class NeuralNetworkDslImpl : NeuralNetworkDsl {

    val modules = mutableListOf<Module>()
    var lastDimension = 0

    fun create() = NetworkBuilder().add(*modules.toTypedArray()).build()

    override fun input(inputSize: Int, id: String) {
        lastDimension = inputSize
        modules.add(Input(Shape(inputSize), name = getDefaultName(id, "Input", modules.size)))
    }

    override fun flatten(id: String, content: FLATTEN.() -> Unit) {
        val impl = FlattenImpl(
            id = getDefaultName(id, "flatten", modules.size)
        )
        impl.content()
        modules += impl.create()
    }

    override fun conv2d(id: String, content: CONV2D.() -> Unit) {
        val impl = Conv2dImpl(
            inChannels = lastDimension,
            id = getDefaultName(id, "conv2d", modules.size)
        )
        impl.content()
        lastDimension = impl.outChannels
        modules += impl.create()
    }

    override fun maxPool2d(id: String, content: MAXPOOL2D.() -> Unit) {
        val impl = MaxPool2dImpl(
            id = getDefaultName(id, "maxPool2d", modules.size)
        )
        impl.content()
        modules += impl.create()
    }

    override fun dense(outputDimension: Int, id: String, content: DENSE.() -> Unit) {
        val inputDimension = lastDimension
        lastDimension = outputDimension
        val impl = DenseImpl(
            inputDimension = inputDimension,
            _outputDimension = outputDimension,
            id = getDefaultName(id, "linear", modules.size)
        )
        impl.content()
        // dense layer consists of linear module and activation function module (2 modules)
        modules += impl.create()
    }

    override fun dense(id: String, content: DENSE.() -> Unit) {
        // This version of dense requires units to be specified in the content block
        val impl = DenseImpl(
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

    override fun activation(id: String, activation: (Tensor) -> Tensor) {
        modules += ActivationsWrapperModule(activation, getDefaultName(id, "activation", modules.size))
    }

    override fun sequential(content: NeuralNetworkDsl.() -> Unit) {
        val sequentialImpl = NeuralNetworkDslImpl()
        sequentialImpl.lastDimension = lastDimension
        sequentialImpl.content()
        lastDimension = sequentialImpl.lastDimension
        modules += sequentialImpl.create()
    }

    override fun stage(id: String, content: NeuralNetworkDsl.() -> Unit) {
        val stageImpl = StageImpl(id)
        stageImpl.lastDimension = lastDimension
        stageImpl.content()
        lastDimension = stageImpl.lastDimension
        modules += stageImpl.create()
    }
}


@NetworkDsl
class NetworkBuilder {
    private val modules = mutableListOf<Module>()

    fun add(vararg modules: Module): NetworkBuilder {
        this.modules += modules.toList()
        return this
    }

    fun build(): Module = MLP(*modules.toTypedArray())
}