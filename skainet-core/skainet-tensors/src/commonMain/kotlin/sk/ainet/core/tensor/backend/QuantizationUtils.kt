package sk.ainet.core.tensor.backend

import sk.ainet.core.tensor.*
import kotlin.math.*

/**
 * Quantization and dequantization utilities for mixed-type tensor operations.
 * 
 * This class provides utilities to convert between different precision types,
 * enabling efficient operations between high-precision and quantized tensors.
 * It supports all DTypes with appropriate scaling and clamping.
 */
public object QuantizationUtils {
    
    /**
     * Quantizes a Float tensor to Int8 with specified scale and zero point.
     */
    public fun quantizeFloatToInt8(
        tensor: Tensor<FP32, Float>,
        scale: Float,
        zeroPoint: Int = 0
    ): Tensor<Int8, Byte> {
        val data = Array<Float>(tensor.shape.volume) { 0f }
        tensor.copyTo(data)
        
        val quantizedData = Array<Byte>(data.size) { i ->
            val scaled = round(data[i] / scale) + zeroPoint
            scaled.coerceIn(-128f, 127f).toInt().toByte()
        }
        
        return BackendDispatcher.createTensor(Int8, tensor.shape, quantizedData)
    }
    
    /**
     * Dequantizes an Int8 tensor to Float with specified scale and zero point.
     */
    public fun dequantizeInt8ToFloat(
        tensor: Tensor<Int8, Byte>,
        scale: Float,
        zeroPoint: Int = 0
    ): Tensor<FP32, Float> {
        val data = Array<Byte>(tensor.shape.volume) { 0 }
        tensor.copyTo(data)
        
        val dequantizedData = Array<Float>(data.size) { i ->
            (data[i].toInt() - zeroPoint) * scale
        }
        
        return BackendDispatcher.createTensor(FP32, tensor.shape, dequantizedData)
    }
    
    /**
     * Quantizes a Float tensor to Int4 with specified scale and zero point.
     */
    public fun quantizeFloatToInt4(
        tensor: Tensor<FP32, Float>,
        scale: Float,
        zeroPoint: Int = 0
    ): Tensor<Int4, Byte> {
        val data = Array<Float>(tensor.shape.volume) { 0f }
        tensor.copyTo(data)
        
        val quantizedData = Array<Byte>(data.size) { i ->
            val scaled = round(data[i] / scale) + zeroPoint
            scaled.coerceIn(-8f, 7f).toInt().toByte()
        }
        
        return BackendDispatcher.createTensor(Int4, tensor.shape, quantizedData)
    }
    
    /**
     * Dequantizes an Int4 tensor to Float with specified scale and zero point.
     */
    public fun dequantizeInt4ToFloat(
        tensor: Tensor<Int4, Byte>,
        scale: Float,
        zeroPoint: Int = 0
    ): Tensor<FP32, Float> {
        val data = Array<Byte>(tensor.shape.volume) { 0 }
        tensor.copyTo(data)
        
        val dequantizedData = Array<Float>(data.size) { i ->
            (data[i].toInt() - zeroPoint) * scale
        }
        
        return BackendDispatcher.createTensor(FP32, tensor.shape, dequantizedData)
    }
    
    /**
     * Quantizes a Float tensor to Ternary (-1, 0, 1) representation.
     */
    public fun quantizeFloatToTernary(
        tensor: Tensor<FP32, Float>,
        threshold: Float = 0.5f
    ): Tensor<Ternary, Byte> {
        val data = Array<Float>(tensor.shape.volume) { 0f }
        tensor.copyTo(data)
        
        val ternaryData = Array<Byte>(data.size) { i ->
            when {
                data[i] > threshold -> 1
                data[i] < -threshold -> -1
                else -> 0
            }
        }
        
        return BackendDispatcher.createTensor(Ternary, tensor.shape, ternaryData)
    }
    
    /**
     * Dequantizes a Ternary tensor to Float representation.
     */
    public fun dequantizeTernaryToFloat(
        tensor: Tensor<Ternary, Byte>,
        scale: Float = 1.0f
    ): Tensor<FP32, Float> {
        val data = Array<Byte>(tensor.shape.volume) { 0 }
        tensor.copyTo(data)
        
        val floatData = Array<Float>(data.size) { i ->
            data[i].toFloat() * scale
        }
        
        return BackendDispatcher.createTensor(FP32, tensor.shape, floatData)
    }
    
    /**
     * Converts between Int32 and Float for mixed precision operations.
     */
    public fun int32ToFloat(tensor: Tensor<Int32, Int>): Tensor<FP32, Float> {
        val data = Array<Int>(tensor.shape.volume) { 0 }
        tensor.copyTo(data)
        
        val floatData = Array<Float>(data.size) { i ->
            data[i].toFloat()
        }
        
        return BackendDispatcher.createTensor(FP32, tensor.shape, floatData)
    }
    
    /**
     * Converts between Float and Int32 with proper rounding.
     */
    public fun floatToInt32(tensor: Tensor<FP32, Float>): Tensor<Int32, Int> {
        val data = Array<Float>(tensor.shape.volume) { 0f }
        tensor.copyTo(data)
        
        val intData = Array<Int>(data.size) { i ->
            round(data[i]).toInt()
        }
        
        return BackendDispatcher.createTensor(Int32, tensor.shape, intData)
    }
    
    /**
     * Converts between FP16 and FP32 (upscaling precision).
     */
    public fun fp16ToFp32(tensor: Tensor<FP16, Float>): Tensor<FP32, Float> {
        val data = Array<Float>(tensor.shape.volume) { 0f }
        tensor.copyTo(data)
        
        // FP16 and FP32 use the same Float type but FP16 has lower precision
        // This is a direct copy since both use Float internally
        return BackendDispatcher.createTensor(FP32, tensor.shape, data)
    }
    
    /**
     * Converts between FP32 and FP16 (downscaling precision).
     */
    public fun fp32ToFp16(tensor: Tensor<FP32, Float>): Tensor<FP16, Float> {
        val data = Array<Float>(tensor.shape.volume) { 0f }
        tensor.copyTo(data)
        
        // For FP16, we would typically need to reduce precision,
        // but since both use Float internally, this is a direct copy
        // In a real implementation, this would apply FP16 precision limits
        return BackendDispatcher.createTensor(FP16, tensor.shape, data)
    }
    
    /**
     * Calculates optimal quantization scale for a tensor.
     */
    public fun calculateQuantizationScale(
        tensor: Tensor<FP32, Float>,
        targetBits: Int
    ): Float {
        val data = Array<Float>(tensor.shape.volume) { 0f }
        tensor.copyTo(data)
        
        val minVal = data.minOrNull() ?: 0f
        val maxVal = data.maxOrNull() ?: 0f
        
        val range = maxOf(abs(minVal), abs(maxVal))
        val maxQuantizedValue = (1 shl (targetBits - 1)) - 1 // e.g., 127 for Int8
        
        return if (range > 0) range / maxQuantizedValue else 1f
    }
    
    /**
     * Packs two Int4 values into a single byte.
     * Lower nibble (bits 0-3): first value
     * Upper nibble (bits 4-7): second value
     * 
     * @param value1 First Int4 value (must be in range -8..7)
     * @param value2 Second Int4 value (must be in range -8..7)
     * @return Packed byte containing both values
     */
    public fun packInt4Values(value1: Byte, value2: Byte): Byte {
        require(value1 in -8..7) { "First Int4 value must be in range -8..7, got $value1" }
        require(value2 in -8..7) { "Second Int4 value must be in range -8..7, got $value2" }
        
        // Convert signed values to 4-bit representation
        val packed1 = value1.toInt() and 0x0F  // Keep lower 4 bits
        val packed2 = (value2.toInt() and 0x0F) shl 4  // Shift to upper 4 bits
        
        return (packed1 or packed2).toByte()
    }
    
    /**
     * Unpacks two Int4 values from a single byte.
     * Lower nibble (bits 0-3): first value
     * Upper nibble (bits 4-7): second value
     * 
     * @param packedByte The byte containing packed Int4 values
     * @return Pair of unpacked Int4 values with proper sign extension
     */
    public fun unpackInt4Values(packedByte: Byte): Pair<Byte, Byte> {
        // Extract lower and upper nibbles
        val value1 = packedByte.toInt() and 0x0F
        val value2 = (packedByte.toInt() shr 4) and 0x0F
        
        // Apply sign extension for 4-bit signed values
        val signExtended1 = if (value1 and 0x08 != 0) value1 or 0xFFFFFFF0.toInt() else value1
        val signExtended2 = if (value2 and 0x08 != 0) value2 or 0xFFFFFFF0.toInt() else value2
        
        return Pair(signExtended1.toByte(), signExtended2.toByte())
    }
    
    /**
     * Packs four Ternary values into a single byte.
     * Each value uses 2 bits: 00=-1, 01=0, 10=1
     * Bits 0-1: fourth value, Bits 2-3: third value, Bits 4-5: second value, Bits 6-7: first value
     * 
     * @param value1 First Ternary value (must be -1, 0, or 1)
     * @param value2 Second Ternary value (must be -1, 0, or 1) 
     * @param value3 Third Ternary value (must be -1, 0, or 1)
     * @param value4 Fourth Ternary value (must be -1, 0, or 1)
     * @return Packed byte containing all four values
     */
    public fun packTernaryValues(value1: Byte, value2: Byte, value3: Byte, value4: Byte): Byte {
        require(value1 in -1..1) { "First Ternary value must be -1, 0, or 1, got $value1" }
        require(value2 in -1..1) { "Second Ternary value must be -1, 0, or 1, got $value2" }
        require(value3 in -1..1) { "Third Ternary value must be -1, 0, or 1, got $value3" }
        require(value4 in -1..1) { "Fourth Ternary value must be -1, 0, or 1, got $value4" }
        
        // Map ternary values to 2-bit representation
        fun ternaryTo2Bit(value: Byte): Int = when (value.toInt()) {
            -1 -> 0  // 00
            0 -> 1   // 01
            1 -> 2   // 10
            else -> throw IllegalArgumentException("Invalid ternary value: $value")
        }
        
        val bits1 = ternaryTo2Bit(value1) shl 6  // Bits 6-7
        val bits2 = ternaryTo2Bit(value2) shl 4  // Bits 4-5
        val bits3 = ternaryTo2Bit(value3) shl 2  // Bits 2-3
        val bits4 = ternaryTo2Bit(value4)        // Bits 0-1
        
        return (bits1 or bits2 or bits3 or bits4).toByte()
    }
    
    /**
     * Unpacks four Ternary values from a single byte.
     * Each value uses 2 bits: 00=-1, 01=0, 10=1
     * Bits 0-1: fourth value, Bits 2-3: third value, Bits 4-5: second value, Bits 6-7: first value
     * 
     * @param packedByte The byte containing packed Ternary values
     * @return Array of four unpacked Ternary values
     */
    public fun unpackTernaryValues(packedByte: Byte): Array<Byte> {
        // Extract 2-bit values
        val bits1 = (packedByte.toInt() shr 6) and 0x03  // Bits 6-7
        val bits2 = (packedByte.toInt() shr 4) and 0x03  // Bits 4-5
        val bits3 = (packedByte.toInt() shr 2) and 0x03  // Bits 2-3
        val bits4 = packedByte.toInt() and 0x03          // Bits 0-1
        
        // Map 2-bit representation back to ternary values
        fun twoBitToTernary(bits: Int): Byte = when (bits) {
            0 -> -1  // 00 -> -1
            1 -> 0   // 01 -> 0
            2 -> 1   // 10 -> 1
            3 -> throw IllegalStateException("Invalid ternary 2-bit value: $bits (11 is reserved)")
            else -> throw IllegalStateException("Unexpected 2-bit value: $bits")
        }
        
        return arrayOf(
            twoBitToTernary(bits1),
            twoBitToTernary(bits2),
            twoBitToTernary(bits3),
            twoBitToTernary(bits4)
        )
    }

    /**
     * Clamps a value to Int8 range with overflow/underflow handling.
     * 
     * @param value The value to clamp
     * @return Value clamped to [-128, 127] range
     */
    public fun clampToInt8(value: Int): Byte = value.coerceIn(-128, 127).toByte()
    
    /**
     * Clamps a value to Int8 range with overflow/underflow handling.
     * 
     * @param value The value to clamp
     * @return Value clamped to [-128, 127] range
     */
    public fun clampToInt8(value: Long): Byte = value.coerceIn(-128L, 127L).toByte()
    
    /**
     * Clamps a value to Int8 range with overflow/underflow handling.
     * 
     * @param value The value to clamp
     * @return Value clamped to [-128, 127] range
     */
    public fun clampToInt8(value: Double): Byte = value.coerceIn(-128.0, 127.0).toInt().toByte()
    
    /**
     * Clamps a value to Int4 range with overflow/underflow handling.
     * 
     * @param value The value to clamp
     * @return Value clamped to [-8, 7] range
     */
    public fun clampToInt4(value: Int): Byte = value.coerceIn(-8, 7).toByte()
    
    /**
     * Clamps a value to Int4 range with overflow/underflow handling.
     * 
     * @param value The value to clamp
     * @return Value clamped to [-8, 7] range
     */
    public fun clampToInt4(value: Long): Byte = value.coerceIn(-8L, 7L).toByte()
    
    /**
     * Clamps a value to Int4 range with overflow/underflow handling.
     * 
     * @param value The value to clamp
     * @return Value clamped to [-8, 7] range
     */
    public fun clampToInt4(value: Double): Byte = value.coerceIn(-8.0, 7.0).toInt().toByte()
    
    /**
     * Clamps a value to Ternary range with overflow/underflow handling.
     * 
     * @param value The value to clamp
     * @return Value clamped to [-1, 1] range
     */
    public fun clampToTernary(value: Int): Byte = value.coerceIn(-1, 1).toByte()
    
    /**
     * Clamps a value to Ternary range with overflow/underflow handling.
     * 
     * @param value The value to clamp
     * @return Value clamped to [-1, 1] range
     */
    public fun clampToTernary(value: Long): Byte = value.coerceIn(-1L, 1L).toByte()
    
    /**
     * Clamps a value to Ternary range with overflow/underflow handling.
     * 
     * @param value The value to clamp
     * @return Value clamped to [-1, 1] range
     */
    public fun clampToTernary(value: Double): Byte = value.coerceIn(-1.0, 1.0).toInt().toByte()
    
    /**
     * Performs safe addition of two quantized values with overflow protection.
     * 
     * @param a First Int8 value
     * @param b Second Int8 value
     * @return Sum clamped to Int8 range
     */
    public fun safeAddInt8(a: Byte, b: Byte): Byte = clampToInt8(a.toInt() + b.toInt())
    
    /**
     * Performs safe subtraction of two quantized values with overflow protection.
     * 
     * @param a First Int8 value
     * @param b Second Int8 value
     * @return Difference clamped to Int8 range
     */
    public fun safeSubInt8(a: Byte, b: Byte): Byte = clampToInt8(a.toInt() - b.toInt())
    
    /**
     * Performs safe multiplication of two quantized values with overflow protection.
     * 
     * @param a First Int8 value
     * @param b Second Int8 value
     * @return Product clamped to Int8 range
     */
    public fun safeMulInt8(a: Byte, b: Byte): Byte = clampToInt8(a.toLong() * b.toLong())
    
    /**
     * Performs safe addition of two Int4 values with overflow protection.
     * 
     * @param a First Int4 value
     * @param b Second Int4 value
     * @return Sum clamped to Int4 range
     */
    public fun safeAddInt4(a: Byte, b: Byte): Byte = clampToInt4(a.toInt() + b.toInt())
    
    /**
     * Performs safe subtraction of two Int4 values with overflow protection.
     * 
     * @param a First Int4 value
     * @param b Second Int4 value
     * @return Difference clamped to Int4 range
     */
    public fun safeSubInt4(a: Byte, b: Byte): Byte = clampToInt4(a.toInt() - b.toInt())
    
    /**
     * Performs safe multiplication of two Int4 values with overflow protection.
     * 
     * @param a First Int4 value
     * @param b Second Int4 value
     * @return Product clamped to Int4 range
     */
    public fun safeMulInt4(a: Byte, b: Byte): Byte = clampToInt4(a.toLong() * b.toLong())
    
    /**
     * Performs safe addition of two Ternary values with overflow protection.
     * 
     * @param a First Ternary value
     * @param b Second Ternary value
     * @return Sum clamped to Ternary range
     */
    public fun safeAddTernary(a: Byte, b: Byte): Byte = clampToTernary(a.toInt() + b.toInt())
    
    /**
     * Performs safe subtraction of two Ternary values with overflow protection.
     * 
     * @param a First Ternary value
     * @param b Second Ternary value
     * @return Difference clamped to Ternary range
     */
    public fun safeSubTernary(a: Byte, b: Byte): Byte = clampToTernary(a.toInt() - b.toInt())
    
    /**
     * Performs safe multiplication of two Ternary values with overflow protection.
     * 
     * @param a First Ternary value
     * @param b Second Ternary value
     * @return Product clamped to Ternary range
     */
    public fun safeMulTernary(a: Byte, b: Byte): Byte = clampToTernary(a.toLong() * b.toLong())

    /**
     * Note: Mixed-precision operations should be performed by explicitly converting
     * tensors using the appropriate quantization/dequantization functions above,
     * then using the BackendDispatcher for operations. This avoids Kotlin's
     * type erasure limitations with generic type checking.
     * 
     * Example usage:
     * val aFloat = dequantizeInt8ToFloat(aTensor, scale)
     * val bFloat = dequantizeInt8ToFloat(bTensor, scale)
     * val result = BackendDispatcher.matmul(aFloat, bFloat)
     */
}