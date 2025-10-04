package sk.ainet.core.tensor.backend

import sk.ainet.core.tensor.*
import kotlin.reflect.KClass

/**
 * Type-safe operation dispatcher that routes operations to the appropriate backend
 * based on the DType of the tensor.
 *
 * This dispatcher provides a unified interface for creating tensors and performing
 * operations while ensuring type safety and optimal backend selection.
 */
public object BackendDispatcher {
    
    private val backends = mutableMapOf<KClass<out DType>, ComputeBackend<*, *, *>>()
    
    /**
     * Registers a backend for a specific DType.
     */
    public fun <T : DType, V> registerBackend(
        dtype: KClass<T>,
        backend: ComputeBackend<T, V, Tensor<T, V>>
    ) {
        backends[dtype] = backend
    }
    
    /**
     * Gets the backend for a specific DType.
     */
    @Suppress("UNCHECKED_CAST")
    public fun <T : DType, V> getBackend(dtype: KClass<T>): ComputeBackend<T, V, Tensor<T, V>>? {
        return backends[dtype] as? ComputeBackend<T, V, Tensor<T, V>>
    }
    
    /**
     * Gets the backend for a specific DType instance.
     */
    @Suppress("UNCHECKED_CAST")
    public fun <T : DType, V> getBackend(dtype: T): ComputeBackend<T, V, Tensor<T, V>>? {
        return backends[dtype::class] as? ComputeBackend<T, V, Tensor<T, V>>
    }
    
    /**
     * Creates a tensor with the specified DType, shape, and data.
     */
    public inline fun <reified T : DType, reified V> createTensor(
        dtype: T,
        shape: Shape,
        data: Array<V>
    ): Tensor<T, V> {
        val backend = getBackend<T, V>(dtype) 
            ?: throw IllegalArgumentException("No backend registered for DType: ${dtype::class.simpleName}")
        
        return CpuTensor.fromArray(shape, data, backend)
    }
    
    /**
     * Creates a tensor filled with zeros.
     */
    public inline fun <reified T : DType, reified V> zeros(
        dtype: T,
        shape: Shape,
        defaultValue: V
    ): Tensor<T, V> {
        val backend = getBackend<T, V>(dtype)
            ?: throw IllegalArgumentException("No backend registered for DType: ${dtype::class.simpleName}")
            
        return if (backend is CpuBackendGeneric<T, V>) {
            backend.zeros(shape)
        } else {
            // Fallback for non-generic backends
            CpuTensor.zeros(shape, backend, defaultValue)
        }
    }
    
    /**
     * Creates a tensor filled with ones.
     */
    public inline fun <reified T : DType, reified V> ones(
        dtype: T,
        shape: Shape,
        oneValue: V
    ): Tensor<T, V> {
        val backend = getBackend<T, V>(dtype)
            ?: throw IllegalArgumentException("No backend registered for DType: ${dtype::class.simpleName}")
            
        return if (backend is CpuBackendGeneric<T, V>) {
            backend.ones(shape)
        } else {
            // Fallback for non-generic backends - create tensor with one value
            CpuTensor.zeros(shape, backend, oneValue)
        }
    }
    
    /**
     * Dispatches a matrix multiplication operation to the appropriate backend.
     */
    public fun <T : DType, V> matmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        return a.ops.matmul(a, b)
    }
    
    /**
     * Dispatches a scaling operation to the appropriate backend.
     */
    public fun <T : DType, V> scale(tensor: Tensor<T, V>, scalar: Double): Tensor<T, V> {
        return tensor.ops.scale(tensor, scalar)
    }
    
    /**
     * Dispatches a dot product operation to the appropriate backend.
     */
    public fun <T : DType, V> dot(a: Tensor<T, V>, b: Tensor<T, V>): Double {
        return a.ops.dot(a, b)
    }
    
    /**
     * Initializes default backend registrations for all supported DTypes.
     */
    public fun initializeDefaultBackends() {
        // Register default backends - this would be called during initialization
        // The specific backends would be registered here based on available implementations
        
        // Example registrations (these would need actual backend implementations):
        // registerBackend(FP32::class, CpuBackendFP32())
        // registerBackend(Int32::class, CpuBackendInt32()) 
        // registerBackend(Int8::class, CpuBackendInt8())
        // registerBackend(FP16::class, CpuBackendFP16())
        // registerBackend(Int4::class, CpuBackendInt4())
        // registerBackend(Ternary::class, CpuBackendTernary())
    }
    
    /**
     * Lists all registered DTypes.
     */
    public fun getRegisteredDTypes(): Set<KClass<out DType>> {
        return backends.keys.toSet()
    }
    
    /**
     * Checks if a DType has a registered backend.
     */
    public fun isRegistered(dtype: KClass<out DType>): Boolean {
        return dtype in backends
    }
    
    /**
     * Unregisters a backend for a specific DType.
     */
    public fun unregisterBackend(dtype: KClass<out DType>) {
        backends.remove(dtype)
    }
    
    /**
     * Clears all registered backends.
     */
    public fun clearAllBackends() {
        backends.clear()
    }
}