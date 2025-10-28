package sk.ainet.lang.nn

import sk.ainet.lang.nn.dsl.NeuralNetworkDsl
import sk.ainet.lang.nn.dsl.NeuralNetworkDslImpl

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32


/**
 * Context for the DSL to define the data type and operations.
 *
 * This class holds the information about the data type and operations
 * that should be used in the DSL. It's used to make the DSL generic
 * and to avoid hardcoding the data type.
 *
 * @param T The default data type.
 */
public interface NeuralNetworkContext<T: DType, V> {

    public val tensorDataFactory: TensorDataFactory
    public val ops: sk.ainet.lang.tensor.ops.TensorOps<V>?

}

/**
 * Creates a context for the DSL with the given configuration.
 *
 * @param T The type of data processed by the modules.
 * @param init The configuration function.
 * @return The configured context.
 */
public fun <T: DType, V> definition(init: NeuralNetworkContext<T, V>.(NeuralNetworkContext<T, V>) -> Module<T, V>): Module<T, V> {
    val instance = DefaultNetworkContext<T, V>()
    return instance.init(instance)
}

public fun <T: DType, V> definition(
    executionContext: ExecutionContext<V>,
    init: NeuralNetworkContext<T, V>.(NeuralNetworkContext<T, V>) -> Module<T, V>
): Module<T, V> {
    val instance = object : NeuralNetworkContext<T, V> {
        override val tensorDataFactory: TensorDataFactory = executionContext.tensorDataFactory
        override val ops: sk.ainet.lang.tensor.ops.TensorOps<V>? = executionContext.ops
    }
    return instance.init(instance)
}

/**
 * Extension function to create a network within a NetworkContext.
 * This bridges the context wrapper with the network DSL using the context's tensor factory.
 */
public inline fun <reified T: DType, V> NeuralNetworkContext<T, V>.network(
    content: NeuralNetworkDsl<T, V>.() -> Unit
): Module<T, V> = NeuralNetworkDslImpl<T, V>(T::class, tensorDataFactory, ops)
    .apply(content)
    .create()

public class DefaultNetworkContext<T : DType, V> : NeuralNetworkContext<T, V> {
    override val tensorDataFactory: TensorDataFactory
        get() = DenseTensorDataFactory()
    override val ops: sk.ainet.lang.tensor.ops.TensorOps<V>?
        get() = null
}