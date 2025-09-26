package sk.ainet.core.tensor.factory

/**
 * Utility object for converting byte arrays to different primitive array types.
 * Supports both little-endian and big-endian byte orders.
 */
public object ByteArrayConverter {
    
    /**
     * Converts a byte array to a FloatArray using little-endian byte order.
     * @param data The source byte array
     * @return FloatArray containing the converted values
     * @throws IllegalArgumentException if the byte array size is not a multiple of 4
     */
    public fun convertBytesToFloatArray(data: ByteArray): FloatArray {
        return convertBytesToFloatArray(data, littleEndian = true)
    }
    
    /**
     * Converts a byte array to a FloatArray with specified endianness.
     * @param data The source byte array
     * @param littleEndian true for little-endian, false for big-endian
     * @return FloatArray containing the converted values
     * @throws IllegalArgumentException if the byte array size is not a multiple of 4
     */
    public fun convertBytesToFloatArray(data: ByteArray, littleEndian: Boolean = true): FloatArray {
        require(data.size % 4 == 0) { 
            "Byte array size (${data.size}) must be a multiple of 4 for float conversion" 
        }
        
        val floatCount = data.size / 4
        val result = FloatArray(floatCount)
        
        for (i in 0 until floatCount) {
            val byteIndex = i * 4
            val intBits = if (littleEndian) {
                // Little-endian: least significant byte first
                (data[byteIndex].toInt() and 0xFF) or
                ((data[byteIndex + 1].toInt() and 0xFF) shl 8) or
                ((data[byteIndex + 2].toInt() and 0xFF) shl 16) or
                ((data[byteIndex + 3].toInt() and 0xFF) shl 24)
            } else {
                // Big-endian: most significant byte first
                ((data[byteIndex].toInt() and 0xFF) shl 24) or
                ((data[byteIndex + 1].toInt() and 0xFF) shl 16) or
                ((data[byteIndex + 2].toInt() and 0xFF) shl 8) or
                (data[byteIndex + 3].toInt() and 0xFF)
            }
            result[i] = Float.fromBits(intBits)
        }
        
        return result
    }
    
    /**
     * Converts a byte array to an IntArray using little-endian byte order.
     * @param data The source byte array
     * @return IntArray containing the converted values
     * @throws IllegalArgumentException if the byte array size is not a multiple of 4
     */
    public fun convertBytesToIntArray(data: ByteArray): IntArray {
        return convertBytesToIntArray(data, littleEndian = true)
    }
    
    /**
     * Converts a byte array to an IntArray with specified endianness.
     * @param data The source byte array
     * @param littleEndian true for little-endian, false for big-endian
     * @return IntArray containing the converted values
     * @throws IllegalArgumentException if the byte array size is not a multiple of 4
     */
    public fun convertBytesToIntArray(data: ByteArray, littleEndian: Boolean = true): IntArray {
        require(data.size % 4 == 0) { 
            "Byte array size (${data.size}) must be a multiple of 4 for int conversion" 
        }
        
        val intCount = data.size / 4
        val result = IntArray(intCount)
        
        for (i in 0 until intCount) {
            val byteIndex = i * 4
            result[i] = if (littleEndian) {
                // Little-endian: least significant byte first
                (data[byteIndex].toInt() and 0xFF) or
                ((data[byteIndex + 1].toInt() and 0xFF) shl 8) or
                ((data[byteIndex + 2].toInt() and 0xFF) shl 16) or
                ((data[byteIndex + 3].toInt() and 0xFF) shl 24)
            } else {
                // Big-endian: most significant byte first
                ((data[byteIndex].toInt() and 0xFF) shl 24) or
                ((data[byteIndex + 1].toInt() and 0xFF) shl 16) or
                ((data[byteIndex + 2].toInt() and 0xFF) shl 8) or
                (data[byteIndex + 3].toInt() and 0xFF)
            }
        }
        
        return result
    }
    
    /**
     * Converts a byte array to a ByteArray (identity function with validation).
     * This is useful for consistency and validation purposes.
     * @param data The source byte array
     * @return A copy of the input ByteArray
     * @throws IllegalArgumentException if the input is null (should not happen in Kotlin)
     */
    public fun convertBytesToByteArray(data: ByteArray): ByteArray {
        return data.copyOf()
    }
    
    /**
     * Validates that a byte array size is appropriate for the given element size.
     * @param data The byte array to validate
     * @param elementSizeBytes The size of each element in bytes (e.g., 4 for Float/Int, 1 for Byte)
     * @param typeName The name of the target type for error messages
     * @throws IllegalArgumentException if the size is not valid
     */
    public fun validateByteArraySize(data: ByteArray, elementSizeBytes: Int, typeName: String) {
        require(data.size % elementSizeBytes == 0) {
            "Byte array size (${data.size}) must be a multiple of $elementSizeBytes for $typeName conversion"
        }
        require(data.isNotEmpty()) {
            "Byte array cannot be empty for $typeName conversion"
        }
    }
    
    /**
     * Gets the expected element count for a byte array given the element size.
     * @param data The byte array
     * @param elementSizeBytes The size of each element in bytes
     * @return The number of elements that can be extracted
     */
    public fun getElementCount(data: ByteArray, elementSizeBytes: Int): Int {
        return data.size / elementSizeBytes
    }
}