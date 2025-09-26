package sk.ainet.core.tensor.factory

import sk.ainet.core.tensor.*
import sk.ainet.core.tensor.backend.CpuTensorInt8
import kotlin.test.*

/**
 * Comprehensive tests for Int8TensorFactory functionality.
 * Tests tensor creation with various dimensions, edge values, and error conditions.
 */
class Int8TensorFactoryTest {

    @BeforeTest
    fun setup() {
        // Setup if needed
    }

    @AfterTest
    fun cleanup() {
        // Cleanup if needed
    }

    @Test
    fun testSimple1DVectorCreation() {
        // Test 1D vector creation (e.g., [1, 2, 3, 4] as bytes)
        val shape = Shape(4)
        val bytes = byteArrayOf(1, 2, 3, 4)
        
        val tensor = Int8TensorFactory.fromGGUFData(shape, bytes)
        
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertTrue(tensor is CpuTensorInt8, "Should be CpuTensorInt8")
        
        // Verify data
        assertEquals(1.toByte(), tensor.get(0), "Value at index 0 should be 1")
        assertEquals(2.toByte(), tensor.get(1), "Value at index 1 should be 2")
        assertEquals(3.toByte(), tensor.get(2), "Value at index 2 should be 3")
        assertEquals(4.toByte(), tensor.get(3), "Value at index 3 should be 4")
    }

    @Test
    fun test2DMatrixCreation() {
        // Test 2D matrix creation (e.g., 3x4 matrix with 12 byte values)
        val shape = Shape(3, 4)
        val bytes = ByteArray(12) { (it + 1).toByte() } // [1, 2, 3, ..., 12]
        
        val tensor = Int8TensorFactory.fromGGUFData(shape, bytes)
        
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(12, tensor.shape.volume, "Volume should be 12")
        
        // Verify some specific values
        assertEquals(1.toByte(), tensor.get(0, 0), "Value at [0,0] should be 1")
        assertEquals(4.toByte(), tensor.get(0, 3), "Value at [0,3] should be 4")
        assertEquals(5.toByte(), tensor.get(1, 0), "Value at [1,0] should be 5")
        assertEquals(12.toByte(), tensor.get(2, 3), "Value at [2,3] should be 12")
    }

    @Test
    fun test3DTensorCreation() {
        // Test 3D tensor creation (e.g., 2x3x4 with 24 byte values)
        val shape = Shape(2, 3, 4)
        val bytes = ByteArray(24) { (it + 1).toByte() } // [1, 2, 3, ..., 24]
        
        val tensor = Int8TensorFactory.fromGGUFData(shape, bytes)
        
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(24, tensor.shape.volume, "Volume should be 24")
        
        // Verify corner values
        assertEquals(1.toByte(), tensor.get(0, 0, 0), "Value at [0,0,0] should be 1")
        assertEquals(24.toByte(), tensor.get(1, 2, 3), "Value at [1,2,3] should be 24")
        assertEquals(13.toByte(), tensor.get(1, 0, 0), "Value at [1,0,0] should be 13")
    }

    @Test
    fun test4DTensorCreation() {
        // Test 4D tensor creation (e.g., 2x2x3x4 with 48 byte values)
        val shape = Shape(2, 2, 3, 4)
        val bytes = ByteArray(48) { (it + 1).toByte() } // [1, 2, 3, ..., 48]
        
        val tensor = Int8TensorFactory.fromGGUFData(shape, bytes)
        
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(48, tensor.shape.volume, "Volume should be 48")
        
        // Verify first and last values
        assertEquals(1.toByte(), tensor.get(0, 0, 0, 0), "First value should be 1")
        assertEquals(48.toByte(), tensor.get(1, 1, 2, 3), "Last value should be 48")
    }

    @Test
    fun testLargeTensorCreation() {
        // Test large tensor creation (e.g., 200x200 matrix with 40,000 bytes)
        val shape = Shape(200, 200)
        val bytes = ByteArray(40000) { (it % 256).toByte() } // Values cycling through byte range
        
        val tensor = Int8TensorFactory.fromGGUFData(shape, bytes)
        
        assertNotNull(tensor, "Large tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(40000, tensor.shape.volume, "Volume should be 40,000")
        
        // Spot check some values
        assertEquals(0.toByte(), tensor.get(0, 0), "First value should be 0")
        assertEquals(199.toByte(), tensor.get(0, 199), "Value at [0,199] should be 199")
        assertEquals((39999 % 256).toByte(), tensor.get(199, 199), "Last value should be correct")
    }

    @Test
    fun testFullByteRange() {
        // Test full byte range (-128 to 127)
        val fullRange = ByteArray(256) { (it - 128).toByte() } // [-128, -127, ..., 127]
        val shape = Shape(256)
        
        val tensor = Int8TensorFactory.fromGGUFData(shape, fullRange)
        
        assertNotNull(tensor, "Tensor with full byte range should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(256, tensor.shape.volume, "Volume should be 256")
        
        // Verify range extremes
        assertEquals((-128).toByte(), tensor.get(0), "First value should be -128")
        assertEquals(127.toByte(), tensor.get(255), "Last value should be 127")
        assertEquals(0.toByte(), tensor.get(128), "Middle value should be 0")
        assertEquals((-1).toByte(), tensor.get(127), "Value at 127 should be -1")
        assertEquals(1.toByte(), tensor.get(129), "Value at 129 should be 1")
    }

    @Test
    fun testErrorConditionMismatchedSizes() {
        // Test error condition: mismatched sizes
        val shape = Shape(3, 4) // Needs 12 bytes
        val wrongSizeBytes = ByteArray(10) // Only 10 bytes
        
        assertFailsWith<IllegalArgumentException>("Should throw for mismatched sizes") {
            Int8TensorFactory.fromGGUFData(shape, wrongSizeBytes)
        }
        
        // Test with too many bytes
        val tooManyBytes = ByteArray(15) // 15 bytes for 12-element tensor
        assertFailsWith<IllegalArgumentException>("Should throw for too many bytes") {
            Int8TensorFactory.fromGGUFData(shape, tooManyBytes)
        }
    }

    @Test
    fun testErrorConditionInvalidShape() {
        // Test error condition: invalid shape
        val bytes = ByteArray(16)
        
        // Test with zero dimension
        assertFailsWith<IllegalArgumentException>("Should throw for zero dimension") {
            val invalidShape = Shape(0, 4)
            Int8TensorFactory.fromGGUFData(invalidShape, bytes)
        }
        
        // Test with negative dimension
        assertFailsWith<IllegalArgumentException>("Should throw for negative dimension") {
            val invalidShape = Shape(-1, 4)
            Int8TensorFactory.fromGGUFData(invalidShape, bytes)
        }
    }

    @Test
    fun testErrorConditionEmptyByteArray() {
        // Test error condition: empty byte array with non-empty shape
        val shape = Shape(2, 2)
        val emptyBytes = ByteArray(0)
        
        assertFailsWith<IllegalArgumentException>("Should throw for empty byte array") {
            Int8TensorFactory.fromGGUFData(shape, emptyBytes)
        }
    }

    @Test
    fun testMinimalTensor() {
        // Test minimal valid tensor (single byte)
        val shape = Shape(1)
        val bytes = byteArrayOf(42)
        
        val tensor = Int8TensorFactory.fromGGUFData(shape, bytes)
        
        assertNotNull(tensor, "Minimal tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(1, tensor.shape.volume, "Volume should be 1")
        assertEquals(42.toByte(), tensor.get(0), "Value should be 42")
    }

    @Test
    fun testSingleDimensionTensors() {
        // Test various single-dimension tensor sizes
        val sizes = intArrayOf(1, 2, 10, 100, 1000)
        
        for (size in sizes) {
            val shape = Shape(size)
            val bytes = ByteArray(size) { (it % 256).toByte() }
            
            val tensor = Int8TensorFactory.fromGGUFData(shape, bytes)
            
            assertNotNull(tensor, "Tensor of size $size should be created")
            assertEquals(shape, tensor.shape, "Shape should match for size $size")
            assertEquals(size, tensor.shape.volume, "Volume should be $size")
            
            // Spot check first and last values
            if (size > 0) {
                assertEquals(0.toByte(), tensor.get(0), "First value should be 0 for size $size")
                if (size > 1) {
                    assertEquals(((size - 1) % 256).toByte(), tensor.get(size - 1), 
                        "Last value should be correct for size $size")
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
            Shape(8, 8, 8) to 512
        )
        
        for ((shape, expectedVolume) in shapeConfigs) {
            val bytes = ByteArray(expectedVolume) { (it + 1).toByte() }
            
            val tensor = Int8TensorFactory.fromGGUFData(shape, bytes)
            
            assertNotNull(tensor, "Tensor with shape $shape should be created")
            assertEquals(shape, tensor.shape, "Shape should match for $shape")
            assertEquals(expectedVolume, tensor.shape.volume, "Volume should be $expectedVolume for $shape")
        }
    }

    @Test
    fun testNegativeValues() {
        // Test tensor with all negative values
        val shape = Shape(4, 4)
        val negativeBytes = ByteArray(16) { (-it - 1).toByte() } // [-1, -2, -3, ..., -16]
        
        val tensor = Int8TensorFactory.fromGGUFData(shape, negativeBytes)
        
        assertNotNull(tensor, "Tensor with negative values should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        
        // Verify negative values
        assertEquals((-1).toByte(), tensor.get(0, 0), "First value should be -1")
        assertEquals((-16).toByte(), tensor.get(3, 3), "Last value should be -16")
        assertEquals((-5).toByte(), tensor.get(1, 0), "Value at [1,0] should be -5")
    }

    @Test
    fun testMixedSignValues() {
        // Test tensor with mixed positive and negative values
        val shape = Shape(8)
        val mixedBytes = byteArrayOf(-128, -64, -32, -1, 0, 1, 32, 127)
        
        val tensor = Int8TensorFactory.fromGGUFData(shape, mixedBytes)
        
        assertNotNull(tensor, "Tensor with mixed sign values should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        
        // Verify mixed values
        assertEquals((-128).toByte(), tensor.get(0), "Value 0 should be -128")
        assertEquals((-64).toByte(), tensor.get(1), "Value 1 should be -64")
        assertEquals((-32).toByte(), tensor.get(2), "Value 2 should be -32")
        assertEquals((-1).toByte(), tensor.get(3), "Value 3 should be -1")
        assertEquals(0.toByte(), tensor.get(4), "Value 4 should be 0")
        assertEquals(1.toByte(), tensor.get(5), "Value 5 should be 1")
        assertEquals(32.toByte(), tensor.get(6), "Value 6 should be 32")
        assertEquals(127.toByte(), tensor.get(7), "Value 7 should be 127")
    }

    @Test
    fun testZeroTensor() {
        // Test tensor filled with zeros
        val shape = Shape(10, 10)
        val zeroBytes = ByteArray(100) { 0 }
        
        val tensor = Int8TensorFactory.fromGGUFData(shape, zeroBytes)
        
        assertNotNull(tensor, "Zero tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        
        // Verify all values are zero
        for (i in 0 until 10) {
            for (j in 0 until 10) {
                assertEquals(0.toByte(), tensor.get(i, j), "Value at [$i,$j] should be 0")
            }
        }
    }

    @Test
    fun testRepeatingPattern() {
        // Test tensor with repeating byte pattern
        val shape = Shape(16)
        val pattern = byteArrayOf(1, 2, 3, 4)
        val bytes = ByteArray(16) { pattern[it % 4] } // Repeat pattern [1,2,3,4] four times
        
        val tensor = Int8TensorFactory.fromGGUFData(shape, bytes)
        
        assertNotNull(tensor, "Pattern tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        
        // Verify repeating pattern
        for (i in 0 until 16) {
            val expectedValue = pattern[i % 4]
            assertEquals(expectedValue, tensor.get(i), "Value at $i should follow pattern")
        }
    }
}