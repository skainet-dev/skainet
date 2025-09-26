package sk.ainet.core.tensor.factory

import sk.ainet.core.tensor.DType
import sk.ainet.core.tensor.FP32
import sk.ainet.core.tensor.Shape
import sk.ainet.core.tensor.Tensor
import sk.ainet.core.tensor.backend.CpuTensorFP32

/**
 * Factory implementation for creating FP32 tensors from byte data.
 * Handles byte-to-float conversion with proper endianness support.
 */
public object FP32TensorFactory : TensorFactoryRegistry.TensorFromBytesFactory<FP32, Float> {
    
    /**
     * Creates a CpuTensorFP32 from GGUF byte data with the specified shape.
     * @param shape The desired shape of the tensor
     * @param data The byte array containing float data in binary format
     * @return A new CpuTensorFP32 instance
     * @throws IllegalArgumentException if data size doesn't match expected float count
     */
    override fun fromGGUFData(shape: Shape, data: ByteArray): Tensor<FP32, Float> {
        // Validate input data size matches shape requirements
        val expectedFloatCount = shape.volume
        val expectedByteSize = expectedFloatCount * 4 // 4 bytes per float
        
        require(data.size == expectedByteSize) {
            "Input data size (${data.size} bytes) does not match expected size for shape $shape " +
            "(expected $expectedByteSize bytes for $expectedFloatCount floats)"
        }
        
        // Convert byte array to float array using little-endian (GGUF standard)
        val floatArray = ByteArrayConverter.convertBytesToFloatArray(data, littleEndian = true)
        
        // Create and return the tensor using the existing factory method
        return CpuTensorFP32.fromArray(shape, floatArray)
    }
    
    /**
     * Creates a tensor from byte data with specified endianness.
     * @param shape The desired shape of the tensor
     * @param data The byte array containing float data
     * @param littleEndian true for little-endian, false for big-endian
     * @return A new CpuTensorFP32 instance
     */
    public fun fromGGUFData(shape: Shape, data: ByteArray, littleEndian: Boolean): Tensor<FP32, Float> {
        // Validate input data size matches shape requirements
        val expectedFloatCount = shape.volume
        val expectedByteSize = expectedFloatCount * 4
        
        require(data.size == expectedByteSize) {
            "Input data size (${data.size} bytes) does not match expected size for shape $shape " +
            "(expected $expectedByteSize bytes for $expectedFloatCount floats)"
        }
        
        // Convert byte array to float array with specified endianness
        val floatArray = ByteArrayConverter.convertBytesToFloatArray(data, littleEndian)
        
        // Create and return the tensor
        return CpuTensorFP32.fromArray(shape, floatArray)
    }
}