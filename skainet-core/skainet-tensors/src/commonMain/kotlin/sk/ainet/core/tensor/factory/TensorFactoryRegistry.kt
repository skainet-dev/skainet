package sk.ainet.core.tensor.factory

import sk.ainet.core.tensor.DType
import sk.ainet.core.tensor.Shape
import sk.ainet.core.tensor.Tensor
import kotlin.reflect.KClass

/**
 * Registry-based factory system for creating tensors from byte data based on DType information.
 */
public object TensorFactoryRegistry {
    
    /**
     * Storage for tensor factories mapped by DType class
     */
    @PublishedApi
    internal val factories: MutableMap<KClass<out DType>, TensorFromBytesFactory<*, *>> = mutableMapOf()
    
    /**
     * Debug logging flag for factory operations (task 4.5)
     */
    public var debugLogging: Boolean = false
    
    /**
     * Interface for factories that can create tensors from byte arrays
     */
    public interface TensorFromBytesFactory<T : DType, V> {
        /**
         * Creates a tensor from byte data with the specified shape
         * @param shape The desired shape of the tensor
         * @param data The byte array containing the tensor data
         * @param littleEndian true for little-endian, false for big-endian
         *
         * @return A new tensor instance
         */
        public fun fromByteArray(shape: Shape, data: ByteArray, littleEndian: Boolean = true): Tensor<T, V>
    }
    
    /**
     * Registers a factory for a specific DType
     * @param factory The factory instance to register
     */
    public inline fun <reified D : DType> registerFactory(factory: TensorFromBytesFactory<D, *>) {
        factories[D::class] = factory
    }
    
    /**
     * Creates a tensor from byte data using the registered factory for the given DType
     * @param dtype The DType instance indicating the tensor type
     * @param shape The desired shape of the tensor
     * @param data The byte array containing the tensor data
     * @return A new tensor instance
     * @throws IllegalArgumentException if no factory is registered for the given DType
     * @throws IllegalArgumentException if shape or data validation fails
     */
    public fun createTensor(dtype: DType, shape: Shape, data: ByteArray): Tensor<*, *> {
        // 4.1: Comprehensive validation for shape-data size matching
        validateShapeAndData(dtype, shape, data)
        
        // 4.2: Proper error messages for unsupported DTypes
        val dtypeClass = dtype::class
        val factory = factories[dtypeClass] 
            ?: throw IllegalArgumentException(
                "No factory registered for DType: ${dtype.name} (${dtypeClass.simpleName}). " +
                "Supported DTypes: ${getRegisteredDTypes().map { it.simpleName }.sortedBy { it }}"
            )
        
        // 4.5: Add logging/debugging support for factory operations
        try {
            @Suppress("UNCHECKED_CAST")
            val result = (factory as TensorFromBytesFactory<DType, Any>).fromByteArray(shape, data)
            
            // Debug logging (can be enabled by setting debugLogging flag)
            if (debugLogging) {
                println("[TensorFactory] Created ${dtype.name} tensor with shape $shape from ${data.size} bytes")
            }
            
            return result
        } catch (e: Exception) {
            // 4.6: Implement graceful handling of corrupted byte data
            when (e) {
                is IllegalArgumentException, is IllegalStateException -> throw e
                else -> throw IllegalArgumentException(
                    "Failed to create ${dtype.name} tensor from byte data: ${e.message}", e
                )
            }
        }
    }
    
    /**
     * Validates shape and data consistency for tensor creation.
     * Implements comprehensive validation as specified in task 4.1, 4.3, and 4.4.
     */
    private fun validateShapeAndData(dtype: DType, shape: Shape, data: ByteArray) {
        // 4.4: Handle edge cases - invalid shapes
        require(shape.dimensions.isNotEmpty()) {
            "Shape cannot be empty. Provided shape: $shape"
        }
        
        require(shape.dimensions.all { it > 0 }) {
            "All shape dimensions must be positive. Provided shape: $shape"
        }
        
        // 4.4: Handle edge cases - null/empty data
        require(data.isNotEmpty() || shape.volume == 0) {
            "Data array cannot be empty for non-zero volume tensor. Shape volume: ${shape.volume}"
        }
        
        // 4.3: Validate byte array alignment requirements for each DType
        val bytesPerElement = when (dtype.sizeInBits) {
            8 -> 1      // Int8, Byte types
            16 -> 2     // FP16
            32 -> 4     // FP32, Int32
            2 -> {      // Ternary: 4 values per byte
                val expectedBytes = (shape.volume + 3) / 4
                require(data.size == expectedBytes) {
                    "Ternary data size mismatch: expected $expectedBytes bytes for ${shape.volume} values, got ${data.size}"
                }
                return // Early return as validation is complete
            }
            4 -> {      // Int4: 2 values per byte
                val expectedBytes = (shape.volume + 1) / 2
                require(data.size == expectedBytes) {
                    "Int4 data size mismatch: expected $expectedBytes bytes for ${shape.volume} values, got ${data.size}"
                }
                return // Early return as validation is complete
            }
            else -> throw IllegalArgumentException(
                "Unsupported DType bit size: ${dtype.sizeInBits} for ${dtype.name}"
            )
        }
        
        // Standard validation for non-packed types
        require(data.size % bytesPerElement == 0) {
            "${dtype.name} data must be aligned to ${bytesPerElement}-byte boundaries. " +
            "Data size: ${data.size}, required alignment: $bytesPerElement bytes"
        }
        
        val expectedBytes = shape.volume * bytesPerElement
        require(data.size == expectedBytes) {
            "Data size mismatch for ${dtype.name}: expected $expectedBytes bytes " +
            "(${shape.volume} values × $bytesPerElement bytes/value), got ${data.size} bytes"
        }
    }
    
    /**
     * Checks if a factory is registered for the given DType
     * @param dtype The DType to check
     * @return true if a factory is registered, false otherwise
     */
    public fun hasFactory(dtype: DType): Boolean {
        return factories.containsKey(dtype::class)
    }
    
    /**
     * Gets all registered DType classes
     * @return Set of registered DType classes
     */
    public fun getRegisteredDTypes(): Set<KClass<out DType>> {
        return factories.keys.toSet()
    }
    
    /**
     * Clears all registered factories (primarily for testing)
     */
    internal fun clearFactories() {
        factories.clear()
    }
}