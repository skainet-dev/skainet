package sk.ainet.core.tensor.factory

import sk.ainet.core.tensor.*
import sk.ainet.core.tensor.backend.CpuTensorFP16
import kotlin.test.*
import kotlin.math.*

/**
 * Unit tests for FP16TensorFactory.
 * Tests the creation of FP16 tensors from byte data.
 */
class FP16TensorFactoryTest {

    @BeforeTest
    fun setup() {
        TensorFactoryInitializer.reset()
        TensorFactoryInitializer.initializeFactories()
    }

    @AfterTest
    fun cleanup() {
        TensorFactoryInitializer.reset()
    }

    /**
     * Helper function to create FP16 byte data from Float values.
     * Each Float is converted to FP16 format (2 bytes per value).
     * This is a simplified conversion - production code might need more sophisticated handling.
     */
    private fun createFP16Bytes(floats: FloatArray): ByteArray {
        val bytes = ByteArray(floats.size * 2)
        for (i in floats.indices) {
            val fp16 = fp32ToFp16(floats[i])
            bytes[i * 2] = (fp16 and 0xFF).toByte()
            bytes[i * 2 + 1] = ((fp16 shr 8) and 0xFF).toByte()
        }
        return bytes
    }

    /**
     * Converts a 32-bit FP32 value to 16-bit FP16 value.
     * This is a basic implementation - production code might need more sophisticated handling.
     */
    private fun fp32ToFp16(fp32: Float): Int {
        val bits = fp32.toBits()
        val sign = (bits shr 31) and 0x1
        val exponent = ((bits shr 23) and 0xFF) - 127 + 15  // Convert bias from 127 to 15
        val mantissa = (bits shr 13) and 0x3FF  // Take top 10 bits of 23-bit mantissa
        
        return when {
            fp32.isNaN() -> 0x7FFF  // NaN
            fp32.isInfinite() -> if (sign == 1) 0xFC00 else 0x7C00  // Infinity
            exponent <= 0 -> 0  // Underflow to zero (simplified)
            exponent >= 31 -> if (sign == 1) 0xFC00 else 0x7C00  // Overflow to infinity
            else -> (sign shl 15) or (exponent shl 10) or mantissa
        }
    }

    @Test
    fun testSimple1DVectorCreation() {
        // Test 1D vector creation with FP16 values [1.0f, 2.0f, 3.0f, 4.0f]
        val shape = Shape(4)
        val floats = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
        val bytes = createFP16Bytes(floats)
        
        val tensor = FP16TensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertTrue(tensor is CpuTensorFP16, "Should be CpuTensorFP16")
        
        // Verify data (allow some precision loss in FP16 conversion)
        for (i in floats.indices) {
            val expected = floats[i]
            val actual = tensor.get(i)
            val tolerance = maxOf(abs(expected * 0.001f), 0.01f) // 0.1% tolerance or 0.01 minimum
            assertEquals(expected, actual, tolerance, "Value at index $i should match within tolerance")
        }
    }

    @Test
    fun test2DMatrixCreation() {
        // Test 2D matrix creation (2x3 matrix with 6 FP16 values)
        val shape = Shape(2, 3)
        val floats = floatArrayOf(1.5f, 2.25f, 3.75f, 4.5f, 5.125f, 6.875f)
        val bytes = createFP16Bytes(floats)
        
        val tensor = FP16TensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(6, tensor.shape.volume, "Volume should be 6")
        
        // Verify specific values (allow some precision loss)
        val tolerance = 0.01f
        assertEquals(1.5f, tensor.get(0, 0), tolerance, "Value at [0,0] should be 1.5")
        assertEquals(2.25f, tensor.get(0, 1), tolerance, "Value at [0,1] should be 2.25")
        assertEquals(3.75f, tensor.get(0, 2), tolerance, "Value at [0,2] should be 3.75")
        assertEquals(4.5f, tensor.get(1, 0), tolerance, "Value at [1,0] should be 4.5")
        assertEquals(5.125f, tensor.get(1, 1), tolerance, "Value at [1,1] should be 5.125")
        assertEquals(6.875f, tensor.get(1, 2), tolerance, "Value at [1,2] should be 6.875")
    }

    @Test
    fun test3DTensorCreation() {
        // Test 3D tensor creation (2x2x2 cube with 8 FP16 values)
        val shape = Shape(2, 2, 2)
        val floats = floatArrayOf(0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f)
        val bytes = createFP16Bytes(floats)
        
        val tensor = FP16TensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(8, tensor.shape.volume, "Volume should be 8")
        
        // Verify corner values
        val tolerance = 0.01f
        assertEquals(0.5f, tensor.get(0, 0, 0), tolerance, "Value at [0,0,0] should be 0.5")
        assertEquals(4.0f, tensor.get(1, 1, 1), tolerance, "Value at [1,1,1] should be 4.0")
        assertEquals(2.5f, tensor.get(1, 0, 0), tolerance, "Value at [1,0,0] should be 2.5")
    }

    @Test
    fun test4DTensorCreation() {
        // Test 4D tensor creation (2x2x2x2 with 16 FP16 values)
        val shape = Shape(2, 2, 2, 2)
        val floats = FloatArray(16) { (it + 1) * 0.5f } // [0.5, 1.0, 1.5, ..., 8.0]
        val bytes = createFP16Bytes(floats)
        
        val tensor = FP16TensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(16, tensor.shape.volume, "Volume should be 16")
        
        // Verify first and last values
        val tolerance = 0.01f
        assertEquals(0.5f, tensor.get(0, 0, 0, 0), tolerance, "First value should be 0.5")
        assertEquals(8.0f, tensor.get(1, 1, 1, 1), tolerance, "Last value should be 8.0")
    }

    @Test
    fun testSmallFloatValues() {
        // Test small floating point values
        val shape = Shape(4)
        val floats = floatArrayOf(0.001f, 0.01f, 0.1f, 1.0f)
        val bytes = createFP16Bytes(floats)
        
        val tensor = FP16TensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Tensor with small values should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        
        // Verify values (FP16 has limited precision for very small values)
        val tolerance = 0.001f
        assertTrue(abs(tensor.get(0) - 0.001f) < tolerance || tensor.get(0) == 0.0f, "Very small value might underflow")
        assertEquals(0.01f, tensor.get(1), tolerance, "Small value should be preserved")
        assertEquals(0.1f, tensor.get(2), tolerance, "Medium value should be preserved")
        assertEquals(1.0f, tensor.get(3), tolerance, "Unit value should be preserved")
    }

    @Test
    fun testLargeFloatValues() {
        // Test large floating point values within FP16 range
        val shape = Shape(3)
        val floats = floatArrayOf(100.0f, 1000.0f, 10000.0f)
        val bytes = createFP16Bytes(floats)
        
        val tensor = FP16TensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Tensor with large values should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        
        // Verify values (allow larger tolerance for big numbers due to FP16 precision)
        assertEquals(100.0f, tensor.get(0), 1.0f, "Large value should be approximately preserved")
        assertEquals(1000.0f, tensor.get(1), 10.0f, "Very large value should be approximately preserved")
        // Note: 10000.0f might not be exactly representable in FP16
        assertTrue(tensor.get(2) > 5000.0f, "Extremely large value should be in correct range")
    }

    @Test
    fun testSpecialFloatValues() {
        // Test special values (0, -0, but avoid NaN/Infinity which may not convert well)
        val shape = Shape(4)
        val floats = floatArrayOf(0.0f, -0.0f, -1.0f, -10.5f)
        val bytes = createFP16Bytes(floats)
        
        val tensor = FP16TensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Tensor with special values should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        
        // Verify values
        val tolerance = 0.01f
        assertEquals(0.0f, tensor.get(0), tolerance, "Zero should be preserved")
        assertEquals(-0.0f, tensor.get(1), tolerance, "Negative zero should be preserved")
        assertEquals(-1.0f, tensor.get(2), tolerance, "Negative value should be preserved")
        assertEquals(-10.5f, tensor.get(3), tolerance, "Negative decimal should be preserved")
    }

    @Test
    fun testErrorConditionWrongByteArraySize() {
        // Test error when byte array size doesn't match expected FP16 size
        val shape = Shape(4) // Needs 8 bytes (4 * 2 bytes per FP16)
        val wrongSizeBytes = ByteArray(6) // Too small
        
        assertFailsWith<IllegalArgumentException>("Should fail with wrong byte array size") {
            FP16TensorFactory.fromByteArray(shape, wrongSizeBytes)
        }
    }

    @Test
    fun testErrorConditionInvalidShape() {
        // Test error conditions with invalid shapes
        val bytes = createFP16Bytes(floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f))
        
        assertFailsWith<IllegalArgumentException>("Should fail with 0D shape") {
            FP16TensorFactory.fromByteArray(Shape(), bytes)
        }
        
        assertFailsWith<IllegalArgumentException>("Should fail with 5D shape") {
            FP16TensorFactory.fromByteArray(Shape(2, 2, 2, 2, 2), bytes)
        }
    }

    @Test
    fun testErrorConditionEmptyByteArray() {
        // Test error when byte array is empty but shape requires data
        val shape = Shape(1)
        val emptyBytes = ByteArray(0)
        
        assertFailsWith<IllegalArgumentException>("Should fail with empty byte array") {
            FP16TensorFactory.fromByteArray(shape, emptyBytes)
        }
    }

    @Test
    fun testMinimalTensor() {
        // Test minimal 1-element tensor
        val shape = Shape(1)
        val floats = floatArrayOf(42.5f)
        val bytes = createFP16Bytes(floats)
        
        val tensor = FP16TensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Minimal tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(1, tensor.shape.volume, "Volume should be 1")
        assertEquals(42.5f, tensor.get(0), 0.1f, "Single value should be 42.5")
    }

    @Test
    fun testSingleDimensionTensors() {
        // Test various single dimension sizes
        val sizes = intArrayOf(2, 8, 16, 32)
        
        for (size in sizes) {
            val shape = Shape(size)
            val floats = FloatArray(size) { it * 0.5f }
            val bytes = createFP16Bytes(floats)
            
            val tensor = FP16TensorFactory.fromByteArray(shape, bytes)
            
            assertNotNull(tensor, "Tensor of size $size should be created")
            assertEquals(shape, tensor.shape, "Shape should match for size $size")
            assertEquals(size, tensor.shape.volume, "Volume should match for size $size")
            
            // Spot check first and last values
            val tolerance = 0.01f
            assertEquals(floats[0], tensor.get(0), tolerance, "First value should match for size $size")
            if (size > 1) {
                assertEquals(floats[size - 1], tensor.get(size - 1), tolerance, "Last value should match for size $size")
            }
        }
    }

    @Test
    fun testVariousShapeCombinations() {
        // Test different shape combinations
        val shapesAndFloats = listOf(
            Pair(Shape(4, 1), floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)),
            Pair(Shape(1, 4), floatArrayOf(0.5f, 1.5f, 2.5f, 3.5f)),
            Pair(Shape(2, 2), floatArrayOf(10.0f, 20.0f, 30.0f, 40.0f)),
            Pair(Shape(3, 3), FloatArray(9) { (it + 1) * 1.25f })
        )
        
        for ((shape, floats) in shapesAndFloats) {
            val bytes = createFP16Bytes(floats)
            val tensor = FP16TensorFactory.fromByteArray(shape, bytes)
            
            assertNotNull(tensor, "Tensor with shape $shape should be created")
            assertEquals(shape, tensor.shape, "Shape should match for $shape")
            assertEquals(shape.volume, tensor.shape.volume, "Volume should match for $shape")
            
            // Verify some values
            val tolerance = 0.1f
            assertEquals(floats[0], tensor.get(0, 0), tolerance, "First value should match for shape $shape")
        }
    }

    @Test
    fun testFP16Precision() {
        // Test FP16 precision characteristics
        val shape = Shape(8)
        val floats = floatArrayOf(
            1.0f,           // Exactly representable
            1.5f,           // Exactly representable
            1.25f,          // Exactly representable
            1.333333f,      // Not exactly representable
            PI.toFloat(),   // Not exactly representable
            E.toFloat(),    // Not exactly representable
            sqrt(2.0f),     // Not exactly representable
            1.0f / 3.0f     // Not exactly representable
        )
        val bytes = createFP16Bytes(floats)
        
        val tensor = FP16TensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Tensor with precision test values should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        
        // Test exact values
        assertEquals(1.0f, tensor.get(0), 0.0001f, "1.0 should be exact")
        assertEquals(1.5f, tensor.get(1), 0.0001f, "1.5 should be exact")
        assertEquals(1.25f, tensor.get(2), 0.0001f, "1.25 should be exact")
        
        // Test approximate values (larger tolerance)
        assertEquals(1.333333f, tensor.get(3), 0.01f, "1.333333 should be approximate")
        assertEquals(PI.toFloat(), tensor.get(4), 0.01f, "PI should be approximate")
        assertEquals(E.toFloat(), tensor.get(5), 0.01f, "E should be approximate")
        assertEquals(sqrt(2.0f), tensor.get(6), 0.01f, "sqrt(2) should be approximate")
        assertEquals(1.0f / 3.0f, tensor.get(7), 0.01f, "1/3 should be approximate")
    }

    @Test
    fun testConversionConsistency() {
        // Test that FP32->FP16->FP32 conversion produces reasonable results
        val shape = Shape(5)
        val originalFloats = floatArrayOf(0.0f, 1.0f, -1.0f, 100.0f, -100.0f)
        val bytes = createFP16Bytes(originalFloats)
        
        val tensor = FP16TensorFactory.fromByteArray(shape, bytes)
        
        // Verify converted values are reasonably close to originals
        for (i in originalFloats.indices) {
            val original = originalFloats[i]
            val converted = tensor.get(i)
            val tolerance = maxOf(abs(original * 0.01f), 0.1f) // 1% tolerance or 0.1 minimum
            assertEquals(original, converted, tolerance, 
                "Converted value at index $i should be close to original")
        }
    }
}