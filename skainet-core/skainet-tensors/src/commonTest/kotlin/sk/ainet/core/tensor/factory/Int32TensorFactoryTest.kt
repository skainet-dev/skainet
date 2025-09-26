package sk.ainet.core.tensor.factory

import sk.ainet.core.tensor.*
import sk.ainet.core.tensor.backend.CpuTensorInt32
import kotlin.test.*

/**
 * Comprehensive tests for Int32TensorFactory functionality.
 * Tests tensor creation with various dimensions, edge values, and error conditions.
 */
class Int32TensorFactoryTest {

    @BeforeTest
    fun setup() {
        // Setup if needed
    }

    @AfterTest
    fun cleanup() {
        // Cleanup if needed
    }

    private fun createIntBytes(ints: IntArray): ByteArray {
        val bytes = ByteArray(ints.size * 4)
        for (i in ints.indices) {
            bytes[i * 4 + 0] = (ints[i] and 0xFF).toByte()
            bytes[i * 4 + 1] = ((ints[i] shr 8) and 0xFF).toByte()
            bytes[i * 4 + 2] = ((ints[i] shr 16) and 0xFF).toByte()
            bytes[i * 4 + 3] = ((ints[i] shr 24) and 0xFF).toByte()
        }
        return bytes
    }

    @Test
    fun testSimple1DVectorCreation() {
        // Test 1D vector creation (e.g., [1, 2, 3, 4] as ints)
        val shape = Shape(4)
        val ints = intArrayOf(1, 2, 3, 4)
        val bytes = createIntBytes(ints)
        
        val tensor = Int32TensorFactory.fromGGUFData(shape, bytes)
        
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertTrue(tensor is CpuTensorInt32, "Should be CpuTensorInt32")
        
        // Verify data
        assertEquals(1, tensor.get(0), "Value at index 0 should be 1")
        assertEquals(2, tensor.get(1), "Value at index 1 should be 2")
        assertEquals(3, tensor.get(2), "Value at index 2 should be 3")
        assertEquals(4, tensor.get(3), "Value at index 3 should be 4")
    }

    @Test
    fun test2DMatrixCreation() {
        // Test 2D matrix creation (e.g., 2x3 matrix with 6 int values)
        val shape = Shape(2, 3)
        val ints = intArrayOf(10, 20, 30, 40, 50, 60)
        val bytes = createIntBytes(ints)
        
        val tensor = Int32TensorFactory.fromGGUFData(shape, bytes)
        
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(6, tensor.shape.volume, "Volume should be 6")
        
        // Verify some specific values
        assertEquals(10, tensor.get(0, 0), "Value at [0,0] should be 10")
        assertEquals(20, tensor.get(0, 1), "Value at [0,1] should be 20")
        assertEquals(30, tensor.get(0, 2), "Value at [0,2] should be 30")
        assertEquals(40, tensor.get(1, 0), "Value at [1,0] should be 40")
        assertEquals(50, tensor.get(1, 1), "Value at [1,1] should be 50")
        assertEquals(60, tensor.get(1, 2), "Value at [1,2] should be 60")
    }

    @Test
    fun test3DTensorCreation() {
        // Test 3D tensor creation (e.g., 2x2x3 with 12 int values)
        val shape = Shape(2, 2, 3)
        val ints = IntArray(12) { (it + 1) * 10 } // [10, 20, 30, ..., 120]
        val bytes = createIntBytes(ints)
        
        val tensor = Int32TensorFactory.fromGGUFData(shape, bytes)
        
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(12, tensor.shape.volume, "Volume should be 12")
        
        // Verify corner values
        assertEquals(10, tensor.get(0, 0, 0), "Value at [0,0,0] should be 10")
        assertEquals(120, tensor.get(1, 1, 2), "Value at [1,1,2] should be 120")
        assertEquals(70, tensor.get(1, 0, 0), "Value at [1,0,0] should be 70")
    }

    @Test
    fun test4DTensorCreation() {
        // Test 4D tensor creation (e.g., 2x2x2x3 with 24 int values)
        val shape = Shape(2, 2, 2, 3)
        val ints = IntArray(24) { it + 100 } // [100, 101, 102, ..., 123]
        val bytes = createIntBytes(ints)
        
        val tensor = Int32TensorFactory.fromGGUFData(shape, bytes)
        
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(24, tensor.shape.volume, "Volume should be 24")
        
        // Verify first and last values
        assertEquals(100, tensor.get(0, 0, 0, 0), "First value should be 100")
        assertEquals(123, tensor.get(1, 1, 1, 2), "Last value should be 123")
    }

    @Test
    fun testLargeTensorCreation() {
        // Test large tensor creation (e.g., 50x50 matrix with 2,500 values)
        val shape = Shape(50, 50)
        val ints = IntArray(2500) { it * 2 } // [0, 2, 4, 6, ..., 4998]
        val bytes = createIntBytes(ints)
        
        val tensor = Int32TensorFactory.fromGGUFData(shape, bytes)
        
        assertNotNull(tensor, "Large tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(2500, tensor.shape.volume, "Volume should be 2,500")
        
        // Spot check some values
        assertEquals(0, tensor.get(0, 0), "First value should be 0")
        assertEquals(98, tensor.get(0, 49), "Value at [0,49] should be 98")
        assertEquals(4998, tensor.get(49, 49), "Last value should be 4998")
    }

    @Test
    fun testExtremeIntValues() {
        // Test extreme int values (Int.MIN_VALUE, Int.MAX_VALUE)
        val extremeInts = intArrayOf(
            Int.MIN_VALUE,
            Int.MAX_VALUE,
            0,
            -1,
            1,
            -1000000,
            1000000,
            -2147483647 // Int.MIN_VALUE + 1
        )
        
        val shape = Shape(8)
        val bytes = createIntBytes(extremeInts)
        
        val tensor = Int32TensorFactory.fromGGUFData(shape, bytes)
        
        assertNotNull(tensor, "Tensor with extreme values should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        
        // Verify extreme values
        assertEquals(Int.MIN_VALUE, tensor.get(0), "First value should be MIN_VALUE")
        assertEquals(Int.MAX_VALUE, tensor.get(1), "Second value should be MAX_VALUE")
        assertEquals(0, tensor.get(2), "Third value should be 0")
        assertEquals(-1, tensor.get(3), "Fourth value should be -1")
        assertEquals(1, tensor.get(4), "Fifth value should be 1")
        assertEquals(-1000000, tensor.get(5), "Sixth value should be -1000000")
        assertEquals(1000000, tensor.get(6), "Seventh value should be 1000000")
        assertEquals(-2147483647, tensor.get(7), "Eighth value should be -2147483647")
    }

    @Test
    fun testByteArrayAlignment() {
        // Test byte array alignment (must be multiples of 4)
        val shape = Shape(3) // 3 ints = 12 bytes
        val validBytes = ByteArray(12) { 0 }
        
        // This should work
        val tensor = Int32TensorFactory.fromGGUFData(shape, validBytes)
        assertNotNull(tensor, "Tensor with properly aligned bytes should be created")
        
        // Test misaligned byte arrays
        val misalignedBytes1 = ByteArray(11) // Not multiple of 4
        assertFailsWith<IllegalArgumentException>("Should throw for 11 bytes") {
            Int32TensorFactory.fromGGUFData(shape, misalignedBytes1)
        }
        
        val misalignedBytes2 = ByteArray(13) // Not multiple of 4
        assertFailsWith<IllegalArgumentException>("Should throw for 13 bytes") {
            Int32TensorFactory.fromGGUFData(shape, misalignedBytes2)
        }
        
        val misalignedBytes3 = ByteArray(10) // Not multiple of 4
        assertFailsWith<IllegalArgumentException>("Should throw for 10 bytes") {
            Int32TensorFactory.fromGGUFData(shape, misalignedBytes3)
        }
    }

    @Test
    fun testErrorConditionWrongByteArraySize() {
        // Test error condition: wrong byte array size
        val shape = Shape(2, 2) // Needs 16 bytes for 4 ints
        val wrongSizeBytes = ByteArray(12) // Only 12 bytes (3 ints)
        
        assertFailsWith<IllegalArgumentException>("Should throw for wrong byte array size") {
            Int32TensorFactory.fromGGUFData(shape, wrongSizeBytes)
        }
        
        // Test with too many bytes
        val tooManyBytes = ByteArray(20) // 20 bytes for 4-int tensor
        assertFailsWith<IllegalArgumentException>("Should throw for too many bytes") {
            Int32TensorFactory.fromGGUFData(shape, tooManyBytes)
        }
    }

    @Test
    fun testErrorConditionInvalidShape() {
        // Test error condition: invalid shape
        val bytes = ByteArray(16) // 4 ints worth of data
        
        // Test with zero dimension
        assertFailsWith<IllegalArgumentException>("Should throw for zero dimension") {
            val invalidShape = Shape(0, 4)
            Int32TensorFactory.fromGGUFData(invalidShape, bytes)
        }
        
        // Test with negative dimension
        assertFailsWith<IllegalArgumentException>("Should throw for negative dimension") {
            val invalidShape = Shape(-1, 4)
            Int32TensorFactory.fromGGUFData(invalidShape, bytes)
        }
    }

    @Test
    fun testErrorConditionEmptyByteArray() {
        // Test error condition: empty byte array with non-empty shape
        val shape = Shape(2, 2)
        val emptyBytes = ByteArray(0)
        
        assertFailsWith<IllegalArgumentException>("Should throw for empty byte array") {
            Int32TensorFactory.fromGGUFData(shape, emptyBytes)
        }
    }

    @Test
    fun testMinimalTensor() {
        // Test minimal valid tensor (single int)
        val shape = Shape(1)
        val ints = intArrayOf(12345)
        val bytes = createIntBytes(ints)
        
        val tensor = Int32TensorFactory.fromGGUFData(shape, bytes)
        
        assertNotNull(tensor, "Minimal tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(1, tensor.shape.volume, "Volume should be 1")
        assertEquals(12345, tensor.get(0), "Value should be 12345")
    }

    @Test
    fun testSingleDimensionTensors() {
        // Test various single-dimension tensor sizes
        val sizes = intArrayOf(1, 2, 10, 100, 500)
        
        for (size in sizes) {
            val shape = Shape(size)
            val ints = IntArray(size) { it * 7 } // Multiply by 7 for varied values
            val bytes = createIntBytes(ints)
            
            val tensor = Int32TensorFactory.fromGGUFData(shape, bytes)
            
            assertNotNull(tensor, "Tensor of size $size should be created")
            assertEquals(shape, tensor.shape, "Shape should match for size $size")
            assertEquals(size, tensor.shape.volume, "Volume should be $size")
            
            // Spot check first and last values
            if (size > 0) {
                assertEquals(0, tensor.get(0), "First value should be 0 for size $size")
                if (size > 1) {
                    assertEquals((size - 1) * 7, tensor.get(size - 1), 
                        "Last value should be ${(size - 1) * 7} for size $size")
                }
            }
        }
    }

    @Test
    fun testVariousShapeCombinations() {
        // Test various valid shape combinations
        val shapeConfigs = listOf(
            Shape(1, 1) to 1,
            Shape(1, 5) to 5,
            Shape(5, 1) to 5,
            Shape(3, 3) to 9,
            Shape(2, 2, 3) to 12,
            Shape(1, 2, 3, 2) to 12,
            Shape(4, 4, 4) to 64
        )
        
        for ((shape, expectedVolume) in shapeConfigs) {
            val ints = IntArray(expectedVolume) { it + 1000 } // Offset by 1000
            val bytes = createIntBytes(ints)
            
            val tensor = Int32TensorFactory.fromGGUFData(shape, bytes)
            
            assertNotNull(tensor, "Tensor with shape $shape should be created")
            assertEquals(shape, tensor.shape, "Shape should match for $shape")
            assertEquals(expectedVolume, tensor.shape.volume, "Volume should be $expectedVolume for $shape")
        }
    }

    @Test
    fun testNegativeValues() {
        // Test tensor with all negative values
        val shape = Shape(3, 3)
        val negativeInts = IntArray(9) { -(it + 1) * 100 } // [-100, -200, -300, ..., -900]
        val bytes = createIntBytes(negativeInts)
        
        val tensor = Int32TensorFactory.fromGGUFData(shape, bytes)
        
        assertNotNull(tensor, "Tensor with negative values should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        
        // Verify negative values
        assertEquals(-100, tensor.get(0, 0), "First value should be -100")
        assertEquals(-900, tensor.get(2, 2), "Last value should be -900")
        assertEquals(-500, tensor.get(1, 1), "Middle value should be -500")
    }

    @Test
    fun testMixedSignValues() {
        // Test tensor with mixed positive and negative values
        val shape = Shape(10)
        val mixedInts = intArrayOf(-1000000, -100, -10, -1, 0, 1, 10, 100, 1000000, Int.MAX_VALUE)
        val bytes = createIntBytes(mixedInts)
        
        val tensor = Int32TensorFactory.fromGGUFData(shape, bytes)
        
        assertNotNull(tensor, "Tensor with mixed sign values should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        
        // Verify mixed values
        for (i in mixedInts.indices) {
            assertEquals(mixedInts[i], tensor.get(i), "Value at $i should match expected")
        }
    }

    @Test
    fun testZeroTensor() {
        // Test tensor filled with zeros
        val shape = Shape(5, 5)
        val zeroInts = IntArray(25) { 0 }
        val bytes = createIntBytes(zeroInts)
        
        val tensor = Int32TensorFactory.fromGGUFData(shape, bytes)
        
        assertNotNull(tensor, "Zero tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        
        // Verify all values are zero
        for (i in 0 until 5) {
            for (j in 0 until 5) {
                assertEquals(0, tensor.get(i, j), "Value at [$i,$j] should be 0")
            }
        }
    }

    @Test
    fun testLargeIntValues() {
        // Test tensor with large integer values
        val largeInts = intArrayOf(
            1000000000,
            -1000000000,
            2000000000,
            -2000000000,
            Int.MAX_VALUE - 1,
            Int.MIN_VALUE + 1
        )
        
        val shape = Shape(6)
        val bytes = createIntBytes(largeInts)
        
        val tensor = Int32TensorFactory.fromGGUFData(shape, bytes)
        
        assertNotNull(tensor, "Tensor with large values should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        
        // Verify large values
        for (i in largeInts.indices) {
            assertEquals(largeInts[i], tensor.get(i), "Large value at $i should match")
        }
    }

    @Test
    fun testSequentialValues() {
        // Test tensor with sequential values
        val shape = Shape(20)
        val sequentialInts = IntArray(20) { it * it } // Square numbers: [0, 1, 4, 9, 16, ...]
        val bytes = createIntBytes(sequentialInts)
        
        val tensor = Int32TensorFactory.fromGGUFData(shape, bytes)
        
        assertNotNull(tensor, "Sequential tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        
        // Verify sequential values
        for (i in 0 until 20) {
            val expected = i * i
            assertEquals(expected, tensor.get(i), "Sequential value at $i should be $expected")
        }
    }

    @Test
    fun testRepeatingPattern() {
        // Test tensor with repeating int pattern
        val shape = Shape(12)
        val pattern = intArrayOf(1, -1, 2, -2)
        val ints = IntArray(12) { pattern[it % 4] } // Repeat pattern [1,-1,2,-2] three times
        val bytes = createIntBytes(ints)
        
        val tensor = Int32TensorFactory.fromGGUFData(shape, bytes)
        
        assertNotNull(tensor, "Pattern tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        
        // Verify repeating pattern
        for (i in 0 until 12) {
            val expectedValue = pattern[i % 4]
            assertEquals(expectedValue, tensor.get(i), "Value at $i should follow pattern")
        }
    }
}