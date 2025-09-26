package sk.ainet.core.tensor.factory

import sk.ainet.core.tensor.Int4
import sk.ainet.core.tensor.Shape
import sk.ainet.core.tensor.Tensor

/**
 * Factory for creating Int4 tensors from byte data.
 * This is currently a placeholder implementation for future development.
 * 
 * Int4 tensors use packed 4-bit integer values, storing 2 values per byte.
 * 
 * TODO: Add TODO implementation for packed 4-bit integer handling
 * TODO: Implement bit unpacking utilities for 4-bit values
 * TODO: Plan storage strategy (2 values per byte)
 */
public object Int4TensorFactory : TensorFactoryRegistry.TensorFromBytesFactory<Int4, Byte> {
    
    /**
     * Creates a tensor from GGUF-style byte data with packed Int4 values.
     * 
     * TODO: This is a placeholder implementation.
     * When CpuTensorInt4 becomes available, this should:
     * 1. Implement bit unpacking for 4-bit values (2 values per byte)
     * 2. Handle signed 4-bit integers (range -8 to 7)
     * 3. Validate input data size matches expected packed size (ceil(shape.volume / 2) bytes)
     * 4. Implement unpacking strategy for accessing individual 4-bit values
     * 5. Create and return proper CpuTensorInt4 instance
     * 
     * Storage strategy considerations:
     * - Each byte contains 2 Int4 values
     * - High nibble (bits 4-7): first value
     * - Low nibble (bits 0-3): second value
     * - Handle odd tensor volumes (last byte may only use high nibble)
     * 
     * @param shape The desired shape of the tensor
     * @param data The byte array containing packed Int4 data (2 values per byte)
     * @return A new Int4 tensor instance
     * @throws NotImplementedError Currently not implemented
     */
    override fun fromByteArray(shape: Shape, data: ByteArray): Tensor<Int4, Byte> {
        // Validate data size for packed Int4 (2 values per byte, rounded up)
        val expectedBytes = (shape.volume + 1) / 2  // Ceiling division for odd volumes
        require(data.size == expectedBytes) {
            "Data size mismatch: expected $expectedBytes bytes for packed Int4 data (${shape.volume} values), got ${data.size}"
        }
        
        // Create CpuTensorInt4 instance with packed data
        return sk.ainet.core.tensor.backend.CpuTensorInt4.fromPackedByteArray(shape, data)
    }
}