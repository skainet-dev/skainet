package sk.ainet.context

import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.dsl.DataContextDsl
import sk.ainet.lang.tensor.dsl.DefinitionContextDslImpl

@DslMarker
public annotation class ContextDsl

@ContextDsl
public interface ContextDslItem

@ContextDsl
public fun data(
    dataFactory: TensorDataFactory = DenseTensorDataFactory(),
    content: DataContextDsl.() -> Unit
) {
    DefinitionContextDslImpl(dataFactory).apply(content)
}
