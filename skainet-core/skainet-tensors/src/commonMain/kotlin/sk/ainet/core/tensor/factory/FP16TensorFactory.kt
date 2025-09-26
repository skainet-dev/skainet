package sk.ainet.core.tensor.factory

import sk.ainet.core.tensor.FP16
import sk.ainet.core.tensor.Shape
import sk.ainet.core.tensor.Tensor

/**
 * Factory for creating FP16 tensors from byte data.
 * This is currently a placeholder implementation for future development.
 * 
 * TODO: Implement proper FP16 to FP32 conversion when CpuTensorFP16 is available
 * TODO: Add conversion utilities for 16-bit float format
 */
public object FP16TensorFactory : TensorFactoryRegistry.TensorFromBytesFactory<FP16, Float> {
    
    /**
     * Creates a tensor from GGUF-style byte data with FP16 values.
     * 
     * TODO: This is a placeholder implementation. 
     * When CpuTensorFP16 becomes available, this should:
     * 1. Convert bytes to FP16 values with proper endianness handling
     * 2. Validate input data size matches expected FP16 count (shape.volume * 2 bytes)
     * 3. Handle FP16 to FP32 conversion if needed
     * 4. Create and return proper CpuTensorFP16 instance
     * 
     * @param shape The desired shape of the tensor
     * @param data The byte array containing FP16 data (2 bytes per value)
     * @return A new FP16 tensor instance
     * @throws NotImplementedError Currently not implemented
     */
    override fun fromByteArray(shape: Shape, data: ByteArray): Tensor<FP16, Float> {
        // Validate data size for FP16 (2 bytes per value)
        val expectedBytes = shape.volume * 2
        require(data.size == expectedBytes) {
            "Data size mismatch: expected $expectedBytes bytes for FP16 data, got ${data.size}"
        }
        
        // Create CpuTensorFP16 instance with FP16 byte data
        return sk.ainet.core.tensor.backend.CpuTensorFP16.fromFP16ByteArray(shape, data)
    }
}