package sk.ainet.core.tensor.factory

import sk.ainet.core.tensor.Ternary
import sk.ainet.core.tensor.Shape
import sk.ainet.core.tensor.Tensor

/**
 * Factory for creating Ternary tensors from byte data.
 * This is currently a placeholder implementation for future development.
 * 
 * Ternary tensors use 2-bit values representing three states: -1, 0, 1.
 * Storage uses bit packing with 4 values per byte.
 * 
 * TODO: Add TODO implementation for ternary value handling (-1, 0, 1)
 * TODO: Implement bit unpacking for 2-bit ternary values
 * TODO: Plan storage strategy (4 values per byte)
 */
public object TernaryTensorFactory : TensorFactoryRegistry.TensorFromBytesFactory<Ternary, Byte> {
    
    /**
     * Creates a tensor from GGUF-style byte data with packed Ternary values.
     * 
     * TODO: This is a placeholder implementation.
     * When CpuTensorTernary becomes available, this should:
     * 1. Implement bit unpacking for 2-bit ternary values (4 values per byte)
     * 2. Handle ternary value mapping (00 -> -1, 01 -> 0, 10 -> 1, 11 -> invalid)
     * 3. Validate input data size matches expected packed size (ceil(shape.volume / 4) bytes)
     * 4. Implement unpacking strategy for accessing individual 2-bit values
     * 5. Create and return proper CpuTensorTernary instance
     * 
     * Storage strategy considerations:
     * - Each byte contains 4 Ternary values using 2 bits each
     * - Bits 6-7: first value
     * - Bits 4-5: second value  
     * - Bits 2-3: third value
     * - Bits 0-1: fourth value
     * - Value encoding: 00 = -1, 01 = 0, 10 = +1, 11 = invalid/reserved
     * - Handle non-multiple-of-4 tensor volumes (unused bits in last byte)
     * 
     * @param shape The desired shape of the tensor
     * @param data The byte array containing packed Ternary data (4 values per byte)
     * @return A new Ternary tensor instance
     * @throws NotImplementedError Currently not implemented
     */
    override fun fromByteArray(shape: Shape, data: ByteArray): Tensor<Ternary, Byte> {
        // Validate data size for packed Ternary (4 values per byte, rounded up)
        val expectedBytes = (shape.volume + 3) / 4  // Ceiling division for non-multiples of 4
        require(data.size == expectedBytes) {
            "Data size mismatch: expected $expectedBytes bytes for packed Ternary data (${shape.volume} values), got ${data.size}"
        }
        
        // Create CpuTensorTernary instance with packed data
        return sk.ainet.core.tensor.backend.CpuTensorTernary.fromPackedByteArray(shape, data)
    }
}