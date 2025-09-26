package sk.ainet.core.tensor.factory

import sk.ainet.core.tensor.*
import sk.ainet.core.tensor.backend.CpuTensorFP32
import kotlin.test.*
import kotlin.math.*

/**
 * Comprehensive tests for FP32TensorFactory functionality.
 * Tests tensor creation with various dimensions, edge values, and error conditions.
 */
class FP32TensorFactoryTest {

    @BeforeTest
    fun setup() {
        // Setup if needed
    }

    @AfterTest
    fun cleanup() {
        // Cleanup if needed
    }

    private fun createFloatBytes(floats: FloatArray): ByteArray {
        val bytes = ByteArray(floats.size * 4)
        for (i in floats.indices) {
            val bits = floats[i].toBits()
            bytes[i * 4 + 0] = (bits and 0xFF).toByte()
            bytes[i * 4 + 1] = ((bits shr 8) and 0xFF).toByte()
            bytes[i * 4 + 2] = ((bits shr 16) and 0xFF).toByte()
            bytes[i * 4 + 3] = ((bits shr 24) and 0xFF).toByte()
        }
        return bytes
    }

    @Test
    fun testSimple1DVectorCreation() {
        // Test 1D vector creation (e.g., [1.0f, 2.0f, 3.0f, 4.0f])
        val shape = Shape(4)
        val floats = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
        val bytes = createFloatBytes(floats)
        
        val tensor = FP32TensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertTrue(tensor is CpuTensorFP32, "Should be CpuTensorFP32")
        
        // Verify data
        for (i in floats.indices) {
            assertEquals(floats[i], tensor.get(i), 0.001f, "Value at index $i should match")
        }
    }

    @Test
    fun test2DMatrixCreation() {
        // Test 2D matrix creation (e.g., 2x3 matrix with 6 float values)
        val shape = Shape(2, 3)
        val floats = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)
        val bytes = createFloatBytes(floats)
        
        val tensor = FP32TensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(6, tensor.shape.volume, "Volume should be 6")
        
        // Verify some specific values
        assertEquals(1.0f, tensor.get(0, 0), 0.001f, "Value at [0,0] should be 1.0")
        assertEquals(2.0f, tensor.get(0, 1), 0.001f, "Value at [0,1] should be 2.0")
        assertEquals(3.0f, tensor.get(0, 2), 0.001f, "Value at [0,2] should be 3.0")
        assertEquals(4.0f, tensor.get(1, 0), 0.001f, "Value at [1,0] should be 4.0")
        assertEquals(5.0f, tensor.get(1, 1), 0.001f, "Value at [1,1] should be 5.0")
        assertEquals(6.0f, tensor.get(1, 2), 0.001f, "Value at [1,2] should be 6.0")
    }

    @Test
    fun test3DTensorCreation() {
        // Test 3D tensor creation (e.g., 2x2x2 cube with 8 float values)
        val shape = Shape(2, 2, 2)
        val floats = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f)
        val bytes = createFloatBytes(floats)
        
        val tensor = FP32TensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(8, tensor.shape.volume, "Volume should be 8")
        
        // Verify corner values
        assertEquals(1.0f, tensor.get(0, 0, 0), 0.001f, "Value at [0,0,0] should be 1.0")
        assertEquals(8.0f, tensor.get(1, 1, 1), 0.001f, "Value at [1,1,1] should be 8.0")
        assertEquals(5.0f, tensor.get(1, 0, 0), 0.001f, "Value at [1,0,0] should be 5.0")
    }

    @Test
    fun test4DTensorCreation() {
        // Test 4D tensor creation (e.g., 2x2x2x2 with 16 float values)
        val shape = Shape(2, 2, 2, 2)
        val floats = FloatArray(16) { (it + 1).toFloat() } // [1.0, 2.0, ..., 16.0]
        val bytes = createFloatBytes(floats)
        
        val tensor = FP32TensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(16, tensor.shape.volume, "Volume should be 16")
        
        // Verify first and last values
        assertEquals(1.0f, tensor.get(0, 0, 0, 0), 0.001f, "First value should be 1.0")
        assertEquals(16.0f, tensor.get(1, 1, 1, 1), 0.001f, "Last value should be 16.0")
    }

    @Test
    fun testLargeTensorCreation() {
        // Test large tensor creation (e.g., 100x100 matrix with 10,000 values)
        val shape = Shape(100, 100)
        val floats = FloatArray(10000) { it.toFloat() }
        val bytes = createFloatBytes(floats)
        
        val tensor = FP32TensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Large tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(10000, tensor.shape.volume, "Volume should be 10,000")
        
        // Spot check some values
        assertEquals(0.0f, tensor.get(0, 0), 0.001f, "First value should be 0.0")
        assertEquals(99.0f, tensor.get(0, 99), 0.001f, "Value at [0,99] should be 99.0")
        assertEquals(9999.0f, tensor.get(99, 99), 0.001f, "Last value should be 9999.0")
    }

    @Test
    fun testEdgeFloatValues() {
        // Test edge values (NaN, infinity, very small/large floats)
        val specialFloats = floatArrayOf(
            Float.NaN,
            Float.POSITIVE_INFINITY,
            Float.NEGATIVE_INFINITY,
            Float.MIN_VALUE,
            Float.MAX_VALUE,
            -Float.MAX_VALUE,
            0.0f,
            -0.0f
        )
        
        val shape = Shape(8)
        val bytes = createFloatBytes(specialFloats)
        
        val tensor = FP32TensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Tensor with special values should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        
        // Verify special values
        assertTrue(tensor.get(0).isNaN(), "First value should be NaN")
        assertTrue(tensor.get(1).isInfinite() && tensor.get(1) > 0, "Second value should be positive infinity")
        assertTrue(tensor.get(2).isInfinite() && tensor.get(2) < 0, "Third value should be negative infinity")
        assertEquals(Float.MIN_VALUE, tensor.get(3), "Fourth value should be MIN_VALUE")
        assertEquals(Float.MAX_VALUE, tensor.get(4), "Fifth value should be MAX_VALUE")
        assertEquals(-Float.MAX_VALUE, tensor.get(5), "Sixth value should be -MAX_VALUE")
        assertEquals(0.0f, tensor.get(6), "Seventh value should be 0.0")
        assertEquals(-0.0f, tensor.get(7), "Eighth value should be -0.0")
    }

    @Test
    fun testErrorConditionWrongByteArraySize() {
        // Test error condition: wrong byte array size
        val shape = Shape(2, 2) // Needs 16 bytes for 4 floats
        val wrongSizeBytes = ByteArray(12) // Only 12 bytes (3 floats)
        
        assertFailsWith<IllegalArgumentException>("Should throw for wrong byte array size") {
            FP32TensorFactory.fromByteArray(shape, wrongSizeBytes)
        }
    }

    @Test
    fun testErrorConditionInvalidShape() {
        // Test error condition: invalid shape
        val bytes = ByteArray(16) // 4 floats worth of data
        
        // Test with zero dimension
        assertFailsWith<IllegalArgumentException>("Should throw for zero dimension") {
            val invalidShape = Shape(0, 4)
            FP32TensorFactory.fromByteArray(invalidShape, bytes)
        }
        
        // Test with negative dimension
        assertFailsWith<IllegalArgumentException>("Should throw for negative dimension") {
            val invalidShape = Shape(-1, 4)
            FP32TensorFactory.fromByteArray(invalidShape, bytes)
        }
    }

    @Test
    fun testErrorConditionEmptyByteArray() {
        // Test error condition: empty byte array with non-empty shape
        val shape = Shape(2, 2)
        val emptyBytes = ByteArray(0)
        
        assertFailsWith<IllegalArgumentException>("Should throw for empty byte array") {
            FP32TensorFactory.fromByteArray(shape, emptyBytes)
        }
    }

    @Test
    fun testErrorConditionMisalignedBytes() {
        // Test error condition: byte array not aligned to 4-byte boundaries
        val shape = Shape(2) // Would need 8 bytes for 2 floats
        val misalignedBytes = ByteArray(7) // Not divisible by 4
        
        assertFailsWith<IllegalArgumentException>("Should throw for misaligned bytes") {
            FP32TensorFactory.fromByteArray(shape, misalignedBytes)
        }
    }

    @Test
    fun testMinimalTensor() {
        // Test minimal valid tensor (1x1 with single float)
        val shape = Shape(1)
        val floats = floatArrayOf(42.0f)
        val bytes = createFloatBytes(floats)
        
        val tensor = FP32TensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Minimal tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(1, tensor.shape.volume, "Volume should be 1")
        assertEquals(42.0f, tensor.get(0), 0.001f, "Value should be 42.0")
    }

    @Test
    fun testSingleDimensionTensors() {
        // Test various single-dimension tensor sizes
        val sizes = intArrayOf(1, 2, 10, 100, 1000)
        
        for (size in sizes) {
            val shape = Shape(size)
            val floats = FloatArray(size) { it.toFloat() }
            val bytes = createFloatBytes(floats)
            
            val tensor = FP32TensorFactory.fromByteArray(shape, bytes)
            
            assertNotNull(tensor, "Tensor of size $size should be created")
            assertEquals(shape, tensor.shape, "Shape should match for size $size")
            assertEquals(size, tensor.shape.volume, "Volume should be $size")
            
            // Spot check first and last values
            if (size > 0) {
                assertEquals(0.0f, tensor.get(0), 0.001f, "First value should be 0.0 for size $size")
                if (size > 1) {
                    assertEquals((size - 1).toFloat(), tensor.get(size - 1), 0.001f, 
                        "Last value should be ${size - 1}.0 for size $size")
                }
            }
        }
    }

    @Test
    fun testVariousShapeCombinations() {
        // Test various valid shape combinations
        val shapeConfigs = listOf(
            Shape(1, 1) to 1,
            Shape(1, 10) to 10,
            Shape(10, 1) to 10,
            Shape(5, 5) to 25,
            Shape(2, 3, 4) to 24,
            Shape(1, 2, 3, 4) to 24,
            Shape(10, 10, 10) to 1000
        )
        
        for ((shape, expectedVolume) in shapeConfigs) {
            val floats = FloatArray(expectedVolume) { (it + 1).toFloat() }
            val bytes = createFloatBytes(floats)
            
            val tensor = FP32TensorFactory.fromByteArray(shape, bytes)
            
            assertNotNull(tensor, "Tensor with shape $shape should be created")
            assertEquals(shape, tensor.shape, "Shape should match for $shape")
            assertEquals(expectedVolume, tensor.shape.volume, "Volume should be $expectedVolume for $shape")
        }
    }

    @Test
    fun testPrecisionMaintenance() {
        // Test that precision is maintained through conversion
        val preciseFloats = floatArrayOf(
            1.234567f,
            -2.345678f,
            0.000001f,
            999999.9f,
            PI.toFloat(),
            E.toFloat()
        )
        
        val shape = Shape(6)
        val bytes = createFloatBytes(preciseFloats)
        
        val tensor = FP32TensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Precision tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        
        // Verify precision is maintained (with some tolerance for float precision)
        for (i in preciseFloats.indices) {
            assertEquals(preciseFloats[i], tensor.get(i), 0.000001f, 
                "Precise value $i should be maintained")
        }
    }
}