package sk.ainet.core.tensor.factory

import sk.ainet.core.tensor.*
import sk.ainet.core.tensor.backend.CpuTensorTernary
import kotlin.test.*

/**
 * Unit tests for TernaryTensorFactory.
 * Tests the creation of Ternary tensors from packed byte data.
 */
class TernaryTensorFactoryTest {

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
     * Helper function to create packed Ternary byte data from ternary values.
     * Each byte contains 4 Ternary values using 2 bits each.
     * Value encoding: -1 -> 00, 0 -> 01, 1 -> 10, invalid -> 11
     */
    private fun createTernaryBytes(values: ByteArray): ByteArray {
        val packedBytes = ByteArray((values.size + 3) / 4)
        for (i in values.indices) {
            val byteIndex = i / 4
            val bitOffset = (i % 4) * 2
            
            // Map ternary value to 2-bit representation
            val bits = when (values[i].toInt()) {
                -1 -> 0  // -1 -> 00
                0 -> 1   // 0 -> 01
                1 -> 2   // 1 -> 10
                else -> throw IllegalArgumentException("Invalid ternary value: ${values[i]}")
            }
            
            // Clear the 2 bits at bitOffset and set new value
            val mask = 0x03 shl bitOffset
            packedBytes[byteIndex] = ((packedBytes[byteIndex].toInt() and mask.inv()) or (bits shl bitOffset)).toByte()
        }
        return packedBytes
    }

    @Test
    fun testSimple1DVectorCreation() {
        // Test 1D vector creation with Ternary values [-1, 0, 1, -1, 0]
        val shape = Shape(5)
        val values = byteArrayOf(-1, 0, 1, -1, 0)
        val bytes = createTernaryBytes(values)
        
        val tensor = TernaryTensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertTrue(tensor is CpuTensorTernary, "Should be CpuTensorTernary")
        
        // Verify data
        for (i in values.indices) {
            assertEquals(values[i].toInt(), tensor.get(i).toInt(), "Value at index $i should match")
        }
    }

    @Test
    fun test2DMatrixCreation() {
        // Test 2D matrix creation (2x3 matrix with 6 Ternary values)
        val shape = Shape(2, 3)
        val values = byteArrayOf(-1, 0, 1, 1, -1, 0)
        val bytes = createTernaryBytes(values)
        
        val tensor = TernaryTensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(6, tensor.shape.volume, "Volume should be 6")
        
        // Verify specific values
        assertEquals(-1, tensor.get(0, 0).toInt(), "Value at [0,0] should be -1")
        assertEquals(0, tensor.get(0, 1).toInt(), "Value at [0,1] should be 0")
        assertEquals(1, tensor.get(0, 2).toInt(), "Value at [0,2] should be 1")
        assertEquals(1, tensor.get(1, 0).toInt(), "Value at [1,0] should be 1")
        assertEquals(-1, tensor.get(1, 1).toInt(), "Value at [1,1] should be -1")
        assertEquals(0, tensor.get(1, 2).toInt(), "Value at [1,2] should be 0")
    }

    @Test
    fun test3DTensorCreation() {
        // Test 3D tensor creation (2x2x2 cube with 8 Ternary values)
        val shape = Shape(2, 2, 2)
        val values = byteArrayOf(-1, 0, 1, -1, 0, 1, -1, 1)
        val bytes = createTernaryBytes(values)
        
        val tensor = TernaryTensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(8, tensor.shape.volume, "Volume should be 8")
        
        // Verify corner values
        assertEquals(-1, tensor.get(0, 0, 0).toInt(), "Value at [0,0,0] should be -1")
        assertEquals(1, tensor.get(1, 1, 1).toInt(), "Value at [1,1,1] should be 1")
        assertEquals(0, tensor.get(1, 0, 0).toInt(), "Value at [1,0,0] should be 0")
    }

    @Test
    fun test4DTensorCreation() {
        // Test 4D tensor creation (2x2x2x2 with 16 Ternary values)
        val shape = Shape(2, 2, 2, 2)
        val values = ByteArray(16) { ((it % 3) - 1).toByte() } // [-1, 0, 1, -1, 0, 1, ...]
        val bytes = createTernaryBytes(values)
        
        val tensor = TernaryTensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(16, tensor.shape.volume, "Volume should be 16")
        
        // Verify first and last values
        assertEquals(-1, tensor.get(0, 0, 0, 0).toInt(), "First value should be -1")
        assertEquals(-1, tensor.get(1, 1, 1, 1).toInt(), "Last value should be -1")
    }

    @Test
    fun testAllTernaryValues() {
        // Test all valid Ternary values [-1, 0, 1] repeated
        val shape = Shape(12)
        val values = ByteArray(12) { ((it % 3) - 1).toByte() } // [-1, 0, 1, -1, 0, 1, ...]
        val bytes = createTernaryBytes(values)
        
        val tensor = TernaryTensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Tensor with all ternary values should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        
        // Verify pattern repeats correctly
        for (i in values.indices) {
            val expectedValue = ((i % 3) - 1).toByte()
            assertEquals(expectedValue.toInt(), tensor.get(i).toInt(), "Value at index $i should be $expectedValue")
        }
    }

    @Test
    fun testNonMultipleOf4Values() {
        // Test non-multiple-of-4 number of values (5 values packed into 2 bytes)
        val shape = Shape(5)
        val values = byteArrayOf(-1, 0, 1, -1, 0)
        val bytes = createTernaryBytes(values)
        
        val tensor = TernaryTensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Tensor with non-multiple-of-4 values should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(5, tensor.shape.volume, "Volume should be 5")
        
        // Verify all values
        for (i in values.indices) {
            assertEquals(values[i].toInt(), tensor.get(i).toInt(), "Value at index $i should match")
        }
    }

    @Test
    fun testErrorConditionWrongByteArraySize() {
        // Test error when byte array size doesn't match expected packed size
        val shape = Shape(5) // Needs 2 bytes (ceil(5/4))
        val wrongSizeBytes = ByteArray(1) // Too small
        
        assertFailsWith<IllegalArgumentException>("Should fail with wrong byte array size") {
            TernaryTensorFactory.fromByteArray(shape, wrongSizeBytes)
        }
    }

    @Test
    fun testErrorConditionInvalidShape() {
        // Test error conditions with invalid shapes
        val bytes = createTernaryBytes(byteArrayOf(1, 0, -1, 1))
        
        assertFailsWith<IllegalArgumentException>("Should fail with 0D shape") {
            TernaryTensorFactory.fromByteArray(Shape(), bytes)
        }
        
        assertFailsWith<IllegalArgumentException>("Should fail with 5D shape") {
            TernaryTensorFactory.fromByteArray(Shape(2, 2, 2, 2, 2), bytes)
        }
    }

    @Test
    fun testErrorConditionEmptyByteArray() {
        // Test error when byte array is empty but shape requires data
        val shape = Shape(1)
        val emptyBytes = ByteArray(0)
        
        assertFailsWith<IllegalArgumentException>("Should fail with empty byte array") {
            TernaryTensorFactory.fromByteArray(shape, emptyBytes)
        }
    }

    @Test
    fun testMinimalTensor() {
        // Test minimal 1-element tensor
        val shape = Shape(1)
        val values = byteArrayOf(1)
        val bytes = createTernaryBytes(values)
        
        val tensor = TernaryTensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Minimal tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(1, tensor.shape.volume, "Volume should be 1")
        assertEquals(1, tensor.get(0).toInt(), "Single value should be 1")
    }

    @Test
    fun testSingleDimensionTensors() {
        // Test various single dimension sizes
        val sizes = intArrayOf(3, 8, 13, 20)
        
        for (size in sizes) {
            val shape = Shape(size)
            val values = ByteArray(size) { ((it % 3) - 1).toByte() }
            val bytes = createTernaryBytes(values)
            
            val tensor = TernaryTensorFactory.fromByteArray(shape, bytes)
            
            assertNotNull(tensor, "Tensor of size $size should be created")
            assertEquals(shape, tensor.shape, "Shape should match for size $size")
            assertEquals(size, tensor.shape.volume, "Volume should match for size $size")
            
            // Spot check first and last values
            assertEquals(values[0].toInt(), tensor.get(0).toInt(), "First value should match for size $size")
            if (size > 1) {
                assertEquals(values[size - 1].toInt(), tensor.get(size - 1).toInt(), "Last value should match for size $size")
            }
        }
    }

    @Test
    fun testVariousShapeCombinations() {
        // Test different shape combinations
        val shapesAndValues = listOf(
            Pair(Shape(4, 1), byteArrayOf(-1, 0, 1, -1)),
            Pair(Shape(1, 4), byteArrayOf(0, 1, -1, 0)),
            Pair(Shape(2, 2), byteArrayOf(1, -1, 0, 1)),
            Pair(Shape(3, 3), ByteArray(9) { ((it % 3) - 1).toByte() })
        )
        
        for ((shape, values) in shapesAndValues) {
            val bytes = createTernaryBytes(values)
            val tensor = TernaryTensorFactory.fromByteArray(shape, bytes)
            
            assertNotNull(tensor, "Tensor with shape $shape should be created")
            assertEquals(shape, tensor.shape, "Shape should match for $shape")
            assertEquals(shape.volume, tensor.shape.volume, "Volume should match for $shape")
            
            // Verify some values
            assertEquals(values[0].toInt(), tensor.get(0, 0).toInt(), "First value should match for shape $shape")
        }
    }

    @Test
    fun testPackingUnpackingConsistency() {
        // Test that packing and unpacking produces consistent results
        val shape = Shape(16) // Full 4-byte packing
        val originalValues = ByteArray(16) { ((it % 3) - 1).toByte() }
        val bytes = createTernaryBytes(originalValues)
        
        val tensor = TernaryTensorFactory.fromByteArray(shape, bytes)
        
        // Verify all values match original
        for (i in originalValues.indices) {
            assertEquals(originalValues[i].toInt(), tensor.get(i).toInt(), 
                "Packed/unpacked value at index $i should match original")
        }
    }

    @Test
    fun testBitPackingCorrectness() {
        // Test specific bit packing patterns to ensure correctness
        val shape = Shape(4)
        val values = byteArrayOf(-1, 0, 1, -1) // Should pack into single byte: 00 01 10 00
        val bytes = createTernaryBytes(values)
        
        assertEquals(1, bytes.size, "Should pack into 1 byte")
        
        val tensor = TernaryTensorFactory.fromByteArray(shape, bytes)
        
        // Verify unpacked values match exactly
        assertEquals(-1, tensor.get(0).toInt(), "First value should be -1")
        assertEquals(0, tensor.get(1).toInt(), "Second value should be 0")
        assertEquals(1, tensor.get(2).toInt(), "Third value should be 1")
        assertEquals(-1, tensor.get(3).toInt(), "Fourth value should be -1")
    }

    @Test
    fun testCreateTernaryBytesHelper() {
        // Test the helper function specifically
        assertFailsWith<IllegalArgumentException>("Should reject invalid ternary value") {
            createTernaryBytes(byteArrayOf(-1, 0, 2)) // 2 is invalid
        }
        
        assertFailsWith<IllegalArgumentException>("Should reject invalid ternary value") {
            createTernaryBytes(byteArrayOf(-2, 0, 1)) // -2 is invalid
        }
    }
}