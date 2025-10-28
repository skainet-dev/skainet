package sk.ainet.execute.context.dsl

import sk.ainet.context.ContextDsl
import sk.ainet.context.ContextDslItem
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.dsl.TensorFactoryContext
import sk.ainet.lang.tensor.operators.withOps
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

@ContextDsl
// Has to remain public so new keyword/block builder can be attached from other libraries
public interface ComputationContextDsl : ContextDslItem {
    // The core method uses an explicit dtype to avoid reified-in-interface
    public fun <T : DType, V> tensor(
        dtype: KClass<T>,
        content: TensorFactoryContext<T, V>.() -> Tensor<T, V>
    ): Tensor<T, V>
}

public inline fun <reified T : DType, V> ComputationContextDsl.tensor(
    noinline content: TensorFactoryContext<T, V>.() -> Tensor<T, V>
): Tensor<T, V> = tensor(T::class, content)

internal class ComputationContextDslImpl<V>(
    private val tensorDataFactory: TensorDataFactory,
    private val ops: sk.ainet.lang.tensor.ops.TensorOps<V>
) : ComputationContextDsl {
    override fun <T : DType, VV> tensor(
        dtype: KClass<T>,
        content: TensorFactoryContext<T, VV>.() -> Tensor<T, VV>
    ): Tensor<T, VV> {
        val ctx = TensorFactoryContext<T, VV>(tensorDataFactory, dtype)
        val t = ctx.content()
        // Bind execution context ops so operator overloads work within computation{}
        @Suppress("UNCHECKED_CAST")
        return (t as Tensor<T, VV>).withOps(ops as sk.ainet.lang.tensor.ops.TensorOps<VV>)
    }
}
