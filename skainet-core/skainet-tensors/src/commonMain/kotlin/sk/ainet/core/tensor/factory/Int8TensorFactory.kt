package sk.ainet.core.tensor.factory

import sk.ainet.core.tensor.Int8
import sk.ainet.core.tensor.Shape
import sk.ainet.core.tensor.Tensor
import sk.ainet.core.tensor.backend.CpuTensorInt8

/**
 * Factory implementation for creating Int8 tensors from byte data.
 * Since Int8 tensors store byte values directly, no conversion is needed.
 */
public object Int8TensorFactory : TensorFactoryRegistry.TensorFromBytesFactory<Int8, Byte> {
    
    /**
     * Creates a CpuTensorInt8 from GGUF byte data with the specified shape.
     * @param shape The desired shape of the tensor
     * @param data The byte array containing the tensor data
     * @return A new CpuTensorInt8 instance
     * @throws IllegalArgumentException if data size doesn't match expected byte count
     */
    override fun fromByteArray(shape: Shape, data: ByteArray): Tensor<Int8, Byte> {
        // Validate input data size matches shape requirements
        val expectedByteCount = shape.volume
        
        require(data.size == expectedByteCount) {
            "Input data size (${data.size} bytes) does not match expected size for shape $shape " +
            "(expected $expectedByteCount bytes)"
        }
        
        // For Int8 tensors, we can use the byte array directly (no conversion needed)
        val byteArray = ByteArrayConverter.convertBytesToByteArray(data)
        
        // Create and return the tensor using the existing factory method
        return CpuTensorInt8.fromArray(shape, byteArray)
    }
    
    /**
     * Creates a tensor from byte data with validation.
     * This method is identical to fromByteArray since Int8 doesn't need endianness conversion.
     * @param shape The desired shape of the tensor
     * @param data The byte array containing the tensor data
     * @return A new CpuTensorInt8 instance
     */
    public fun fromByteData(shape: Shape, data: ByteArray): Tensor<Int8, Byte> {
        return fromByteArray(shape, data)
    }
    
    /**
     * Validates that the input data size matches the expected tensor volume.
     * @param shape The tensor shape
     * @param data The input data
     * @throws IllegalArgumentException if validation fails
     */
    private fun validateInput(shape: Shape, data: ByteArray) {
        val expectedByteCount = shape.volume
        require(data.size == expectedByteCount) {
            "Input data size (${data.size} bytes) does not match expected size for shape $shape " +
            "(expected $expectedByteCount bytes)"
        }
        require(data.isNotEmpty()) {
            "Input data cannot be empty"
        }
    }
}