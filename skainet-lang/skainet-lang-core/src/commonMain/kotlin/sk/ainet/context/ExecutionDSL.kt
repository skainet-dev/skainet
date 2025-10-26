package sk.ainet.context

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.tensor.operators.withOps
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.DType

/**
 * Holds the current execution context for the active DSL block.
 * Note: This is a simple global holder intended for single-threaded test/demo scenarios.
 */
public object ExecutionEnvironment {
    @PublishedApi
    internal var current: ExecutionContext<*>? = null
}

/**
 * Executes the given block within the specified execution context.
 * Any tensors created via the tensor DSL inside this block will use
 * the context's TensorDataFactory and be bound to the context's ops.
 */
public inline fun <V> execute(context: ExecutionContext<V>, block: () -> Unit) {
    val prev = ExecutionEnvironment.current
    ExecutionEnvironment.current = context
    try {
        block()
    } finally {
        ExecutionEnvironment.current = prev
    }
}

/**
 * Context-aware tensor builder that uses the current ExecutionContext if present.
 * This overload shadows the default-factory version to prefer the active context.
 */
public inline fun <reified T : DType, V> tensor(
    noinline content: sk.ainet.lang.tensor.dsl.TensorFactoryContext<T, V>.() -> Tensor<T, V>
): Tensor<T, V> {
    val ctx = ExecutionEnvironment.current
    return if (ctx != null) {
        @Suppress("UNCHECKED_CAST")
        val typedCtx = ctx as ExecutionContext<V>
        val t = sk.ainet.lang.tensor.dsl.tensor<T, V>(typedCtx.tensorDataFactory, content)
        t.withOps(typedCtx.ops as TensorOps<V>)
    } else {
        // fallback to default behavior when no execution context is active
        sk.ainet.lang.tensor.dsl.tensor<T, V>(content)
    }
}
