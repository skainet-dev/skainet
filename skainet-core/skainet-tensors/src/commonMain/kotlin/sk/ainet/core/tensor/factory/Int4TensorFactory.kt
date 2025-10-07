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
 */
public object Int4TensorFactory : TensorFactoryRegistry.TensorFromBytesFactory<Int4, Byte> {

    /**
     * Creates a tensor from GGUF-style byte data with packed Int4 values.
     *
     * Int4 tensor creation is currently not implemented.
     * This is a placeholder for future development.
     *
     * @param shape The desired shape of the tensor
     * @param data The byte array containing packed Int4 data (2 values per byte)
     * @param littleEndian true for little-endian, false for big-endian
     * @return A new Int4 tensor instance
     * @throws NotImplementedError Currently not implemented
     */
    override fun fromByteArray(shape: Shape, data: ByteArray, littleEndian: Boolean): Tensor<Int4, Byte> {
        throw NotImplementedError("Int4 tensor creation is not yet implemented")
    }
}