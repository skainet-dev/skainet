package sk.ainet.lang.nn.dsl

import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.types.DType

public class DefaultNetworkContext<T : DType, V> : NetworkContext<T, V> {
    override val tensorDataFactory: TensorDataFactory
        get() = DenseTensorDataFactory()
}