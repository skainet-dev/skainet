package sk.ainet.execute.context

import sk.ainet.context.ContextDsl
import sk.ainet.context.ExecutionContext
import sk.ainet.execute.context.dsl.ComputationContextDsl
import sk.ainet.execute.context.dsl.ComputationContextDslImpl
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.TensorDataFactory


@ContextDsl
public fun <V> computation(
    executionContext: ExecutionContext<V>,
    dataFactory: TensorDataFactory = executionContext.tensorDataFactory,
    content: ComputationContextDsl.() -> Unit
) {
    val dsl = sk.ainet.execute.context.dsl.ComputationContextDslImpl<V>(
        tensorDataFactory = dataFactory,
        ops = executionContext.ops
    )
    dsl.apply(content)
}

public fun interface ComputationBlockWithContext<V> {
    public fun ComputationContextDsl.invoke(computation: ExecutionContext<V>)
}

@ContextDsl
public fun <V> computation(
    executionContext: ExecutionContext<V>,
    dataFactory: TensorDataFactory = executionContext.tensorDataFactory,
    content: ComputationBlockWithContext<V>
) {
    val dsl = sk.ainet.execute.context.dsl.ComputationContextDslImpl<V>(
        tensorDataFactory = dataFactory,
        ops = executionContext.ops
    )
    content.run { dsl.invoke(executionContext) }
}
