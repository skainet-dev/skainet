package sk.ainet.core.tensor.factory

import sk.ainet.core.tensor.Int32
import sk.ainet.core.tensor.Shape
import sk.ainet.core.tensor.Tensor
import sk.ainet.core.tensor.backend.CpuTensorInt32

/**
 * Factory implementation for creating Int32 tensors from byte data.
 * Handles byte-to-int conversion with proper endianness support.
 */
public object Int32TensorFactory : TensorFactoryRegistry.TensorFromBytesFactory<Int32, Int> {
    
    /**
     * Creates a CpuTensorInt32 from GGUF byte data with the specified shape.
     * @param shape The desired shape of the tensor
     * @param data The byte array containing int data in binary format
     * @return A new CpuTensorInt32 instance
     * @throws IllegalArgumentException if data size doesn't match expected int count
     */
    override fun fromByteArray(shape: Shape, data: ByteArray): Tensor<Int32, Int> {
        // Validate input data size matches shape requirements
        val expectedIntCount = shape.volume
        val expectedByteSize = expectedIntCount * 4 // 4 bytes per int
        
        require(data.size == expectedByteSize) {
            "Input data size (${data.size} bytes) does not match expected size for shape $shape " +
            "(expected $expectedByteSize bytes for $expectedIntCount ints)"
        }
        
        // Convert byte array to int array using little-endian (GGUF standard)
        val intArray = ByteArrayConverter.convertBytesToIntArray(data, littleEndian = true)
        
        // Create and return the tensor using the existing factory method
        return CpuTensorInt32.fromArray(shape, intArray)
    }
    
    /**
     * Creates a tensor from byte data with specified endianness.
     * @param shape The desired shape of the tensor
     * @param data The byte array containing int data
     * @param littleEndian true for little-endian, false for big-endian
     * @return A new CpuTensorInt32 instance
     */
    public fun fromGGUFData(shape: Shape, data: ByteArray, littleEndian: Boolean): Tensor<Int32, Int> {
        // Validate input data size matches shape requirements
        val expectedIntCount = shape.volume
        val expectedByteSize = expectedIntCount * 4
        
        require(data.size == expectedByteSize) {
            "Input data size (${data.size} bytes) does not match expected size for shape $shape " +
            "(expected $expectedByteSize bytes for $expectedIntCount ints)"
        }
        
        // Convert byte array to int array with specified endianness
        val intArray = ByteArrayConverter.convertBytesToIntArray(data, littleEndian)
        
        // Create and return the tensor
        return CpuTensorInt32.fromArray(shape, intArray)
    }
    
    /**
     * Validates that the input data is properly aligned for int conversion.
     * @param shape The tensor shape
     * @param data The input data
     * @throws IllegalArgumentException if validation fails
     */
    private fun validateInput(shape: Shape, data: ByteArray) {
        val expectedIntCount = shape.volume
        val expectedByteSize = expectedIntCount * 4
        
        require(data.size == expectedByteSize) {
            "Input data size (${data.size} bytes) does not match expected size for shape $shape " +
            "(expected $expectedByteSize bytes for $expectedIntCount ints)"
        }
        require(data.isNotEmpty()) {
            "Input data cannot be empty"
        }
        require(data.size % 4 == 0) {
            "Input data size (${data.size} bytes) must be a multiple of 4 for int conversion"
        }
    }
}