package sk.ainet.lang.nn.dsl

import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.types.DType


/**
 * Context for the DSL to define the data type and operations.
 *
 * This class holds the information about the data type and operations
 * that should be used in the DSL. It's used to make the DSL generic
 * and to avoid hardcoding the data type.
 *
 * @param T The default data type.
 */
public interface NetworkContext<T: DType, V> {

    public val tensorDataFactory: TensorDataFactory

}

/**
 * Creates a context for the DSL with the given configuration.
 *
 * @param T The type of data processed by the modules.
 * @param init The configuration function.
 * @return The configured context.
 */
public fun <T: DType, V> context(init: NetworkContext<T, V>.(NetworkContext<T, V>) -> Module<T, V>): Module<T, V> {
    val instance = DefaultNetworkContext<T, V>()
    return instance.init(instance)
}

/**
 * Extension function to create a network within a NetworkContext.
 * This bridges the context wrapper with the network DSL using the context's tensor factory.
 */
public inline fun <reified T: DType, V> NetworkContext<T, V>.network(
    content: NeuralNetworkDsl<T, V>.() -> Unit
): Module<T, V> = network(tensorDataFactory, content)