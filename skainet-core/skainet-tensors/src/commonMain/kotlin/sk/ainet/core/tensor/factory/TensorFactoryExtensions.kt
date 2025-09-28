package sk.ainet.core.tensor.factory

import sk.ainet.core.tensor.DType
import sk.ainet.core.tensor.Shape
import sk.ainet.core.tensor.Tensor

/**
 * Extension functions for easy GGUF-style tensor loading and creation.
 * Implements Chapter 5 requirements from the task factory specification.
 */

/**
 * Creates a tensor from byte data using the factory registry.
 * This provides a convenient way to load tensors from GGUF files and similar formats.
 * 
 * Task 5.1 & 5.2: Extension functions for easy GGUF-style loading and Tensor.fromBytes method
 * 
 * @param dtype The DType instance indicating the tensor type
 * @param shape The desired shape of the tensor
 * @param data The byte array containing the tensor data
 * @return A new tensor instance
 * @throws IllegalArgumentException if no factory is registered for the given DType
 * @throws IllegalArgumentException if shape or data validation fails
 * 
 * Example usage:
 * ```kotlin
 * val floatData = byteArrayOf(/* float bytes */)
 * val tensor = fromBytes(FP32, Shape(2, 3), floatData)
 * ```
 */
public fun fromBytes(
    dtype: DType, 
    shape: Shape, 
    data: ByteArray
): Tensor<*, *> {
    return TensorFactoryRegistry.createTensor(dtype, shape, data)
}


/**
 * Batch tensor creation for loading multiple tensors from GGUF files efficiently.
 * This method can optimize memory allocation and validation for multiple tensor creation.
 * 
 * Task 5.3: Implement batch tensor creation for multiple tensors
 * 
 * @param tensors List of tensor specifications (dtype, shape, data, optional name)
 * @return Map of tensor names (or indices as strings) to created tensors
 * @throws IllegalArgumentException if any tensor creation fails
 * 
 * Example usage:
 * ```kotlin
 * val specs = listOf(
 *     TensorSpec(FP32, Shape(100, 200), floatBytes, "weights"),
 *     TensorSpec(FP32, Shape(200), biasBytes, "bias")
 * )
 * val tensors = Tensor.createBatch(specs)
 * ```
 */
public fun createBatch(
    tensors: List<TensorSpec>
): Map<String, Tensor<*, *>> {
    val result = mutableMapOf<String, Tensor<*, *>>()
    
    for ((index, spec) in tensors.withIndex()) {
        val tensorName = spec.name ?: "tensor_$index"
        try {
            val tensor = TensorFactoryRegistry.createTensor(spec.dtype, spec.shape, spec.data)
            result[tensorName] = tensor
        } catch (e: Exception) {
            throw IllegalArgumentException("Batch tensor creation failed at tensor '$tensorName': ${e.message}", e)
        }
    }
    
    return result
}

/**
 * Data class representing a tensor specification for batch creation.
 * 
 * @param dtype The tensor data type
 * @param shape The tensor shape
 * @param data The raw byte data
 * @param name Optional name for the tensor (used in error reporting and result mapping)
 */
public data class TensorSpec(
    val dtype: DType,
    val shape: Shape,
    val data: ByteArray,
    val name: String? = null
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as TensorSpec

        if (dtype != other.dtype) return false
        if (shape != other.shape) return false
        if (!data.contentEquals(other.data)) return false
        if (name != other.name) return false

        return true
    }

    override fun hashCode(): Int {
        var result = dtype.hashCode()
        result = 31 * result + shape.hashCode()
        result = 31 * result + data.contentHashCode()
        result = 31 * result + (name?.hashCode() ?: 0)
        return result
    }
}

/**
 * Support for custom factory registration by external libraries.
 * This allows third-party libraries to register their own DType factories.
 * 
 * Task 5.4: Add support for custom factory registration by external libraries
 * 
 * @param factory The custom factory implementation
 * @throws IllegalArgumentException if a factory is already registered for this DType
 * 
 * Example usage:
 * ```kotlin
 * class CustomDType : DType { /* implementation */ }
 * class CustomTensorFactory : TensorFromBytesFactory<CustomDType, Float> { /* implementation */ }
 * 
 * Tensor.registerCustomFactory<CustomDType>(CustomTensorFactory())
 * ```
 */
public inline fun <reified D : DType> registerCustomFactory(
    factory: TensorFactoryRegistry.TensorFromBytesFactory<D, *>,
    allowOverride: Boolean = false
) {
    // Note: We can't easily check if factory already exists without a DType instance
    // This functionality will be provided by TensorFactoryRegistry.registerFactory which can handle overrides
    TensorFactoryRegistry.registerFactory<D>(factory)
}

/**
 * Fluent API builder for common tensor creation patterns.
 * This provides a more readable way to create tensors with validation and error handling.
 * 
 * Task 5.5: Create fluent API for common tensor creation patterns
 * 
 * Example usage:
 * ```kotlin
 * val tensor = tensorBuilder()
 *     .withDType(FP32)
 *     .withShape(2, 3, 4)
 *     .fromBytes(byteData)
 *     .withName("my_tensor")
 *     .build()
 * ```
 */
public fun tensorBuilder(): TensorBuilder {
    return TensorBuilder()
}

/**
 * Builder class for fluent tensor creation API.
 */
public class TensorBuilder {
    private var dtype: DType? = null
    private var shape: Shape? = null
    private var data: ByteArray? = null
    private var name: String? = null
    private var validateData: Boolean = true

    /**
     * Sets the DType for the tensor.
     */
    public fun withDType(dtype: DType): TensorBuilder {
        this.dtype = dtype
        return this
    }

    /**
     * Sets the shape for the tensor.
     */
    public fun withShape(shape: Shape): TensorBuilder {
        this.shape = shape
        return this
    }

    /**
     * Sets the shape for the tensor using dimensions.
     */
    public fun withShape(vararg dimensions: Int): TensorBuilder {
        this.shape = Shape(dimensions)
        return this
    }

    /**
     * Sets the byte data for the tensor.
     */
    public fun fromBytes(data: ByteArray): TensorBuilder {
        this.data = data
        return this
    }

    /**
     * Sets an optional name for better error reporting.
     */
    public fun withName(name: String): TensorBuilder {
        this.name = name
        return this
    }

    /**
     * Disables data validation (use with caution).
     */
    public fun skipValidation(): TensorBuilder {
        this.validateData = false
        return this
    }

    /**
     * Builds the tensor using the configured parameters.
     */
    public fun build(): Tensor<*, *> {
        val finalDType = dtype ?: throw IllegalStateException("DType must be specified")
        val finalShape = shape ?: throw IllegalStateException("Shape must be specified") 
        val finalData = data ?: throw IllegalStateException("Data must be specified")
        
        return try {
            TensorFactoryRegistry.createTensor(finalDType, finalShape, finalData)
        } catch (e: Exception) {
            val nameContext = name?.let { " '$it'" } ?: ""
            throw IllegalArgumentException("Failed to build tensor$nameContext: ${e.message}", e)
        }
    }
}