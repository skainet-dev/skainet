package sk.ainet.lang.tensor.ops

import sk.ainet.lang.types.DType

/**
 * Minimal shim to provide a current graph context for operator extensions.
 * This avoids compile errors while full graph context is under development.
 */
public data class SimpleGraphContext<T: DType, V>(
    val graphOps: TensorOps<V>
)

@Suppress("UNCHECKED_CAST")
public fun <T: DType, V> getCurrentGraphContext(): SimpleGraphContext<T, V> {
    // Fallback to no-op ops that only validate shapes and return zero-filled tensors
    return SimpleGraphContext(VoidTensorOps<V>())
}
