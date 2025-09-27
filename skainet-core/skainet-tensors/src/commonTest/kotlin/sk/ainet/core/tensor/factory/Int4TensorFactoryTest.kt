package sk.ainet.core.tensor.factory

import sk.ainet.core.tensor.*
import sk.ainet.core.tensor.backend.CpuTensorInt4
import kotlin.test.*

/**
 * Unit tests for Int4TensorFactory.
 * Tests the creation of Int4 tensors from packed byte data.
 */
class Int4TensorFactoryTest {

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
     * Helper function to create packed Int4 byte data from signed 4-bit values.
     * Each byte contains 2 Int4 values: high nibble (first value) and low nibble (second value).
     * Values are clamped to [-8, 7] range for signed 4-bit integers.
     */
    private fun createInt4Bytes(values: ByteArray): ByteArray {
        val packedBytes = ByteArray((values.size + 1) / 2)
        for (i in values.indices) {
            val clampedValue = values[i].toInt().coerceIn(-8, 7)
            val nibble = if (clampedValue < 0) clampedValue + 16 else clampedValue
            val byteIndex = i / 2
            val isHighNibble = (i % 2) == 0
            
            if (isHighNibble) {
                // Set high nibble, preserve low nibble
                packedBytes[byteIndex] = ((nibble shl 4) or (packedBytes[byteIndex].toInt() and 0x0F)).toByte()
            } else {
                // Set low nibble, preserve high nibble
                packedBytes[byteIndex] = ((packedBytes[byteIndex].toInt() and 0xF0) or nibble).toByte()
            }
        }
        return packedBytes
    }

    @Test
    fun testSimple1DVectorCreation() {
        // Test 1D vector creation with Int4 values [-8, -1, 0, 1, 7]
        val shape = Shape(5)
        val values = byteArrayOf(-8, -1, 0, 1, 7)
        val bytes = createInt4Bytes(values)
        
        val tensor = Int4TensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertTrue(tensor is CpuTensorInt4, "Should be CpuTensorInt4")
        
        // Verify data
        for (i in values.indices) {
            assertEquals(values[i], tensor.get(i), "Value at index $i should match")
        }
    }

    @Test
    fun test2DMatrixCreation() {
        // Test 2D matrix creation (2x3 matrix with 6 Int4 values)
        val shape = Shape(2, 3)
        val values = byteArrayOf(-8, -4, 0, 3, 7, -1)
        val bytes = createInt4Bytes(values)
        
        val tensor = Int4TensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(6, tensor.shape.volume, "Volume should be 6")
        
        // Verify specific values
        assertEquals(-8, tensor.get(0, 0).toInt(), "Value at [0,0] should be -8")
        assertEquals(-4, tensor.get(0, 1).toInt(), "Value at [0,1] should be -4")
        assertEquals(0, tensor.get(0, 2).toInt(), "Value at [0,2] should be 0")
        assertEquals(3, tensor.get(1, 0).toInt(), "Value at [1,0] should be 3")
        assertEquals(7, tensor.get(1, 1).toInt(), "Value at [1,1] should be 7")
        assertEquals(-1, tensor.get(1, 2).toInt(), "Value at [1,2] should be -1")
    }

    @Test
    fun test3DTensorCreation() {
        // Test 3D tensor creation (2x2x2 cube with 8 Int4 values)
        val shape = Shape(2, 2, 2)
        val values = byteArrayOf(-8, -4, -2, 0, 1, 3, 5, 7)
        val bytes = createInt4Bytes(values)
        
        val tensor = Int4TensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(8, tensor.shape.volume, "Volume should be 8")
        
        // Verify corner values
        assertEquals(-8, tensor.get(0, 0, 0).toInt(), "Value at [0,0,0] should be -8")
        assertEquals(7, tensor.get(1, 1, 1).toInt(), "Value at [1,1,1] should be 7")
        assertEquals(1, tensor.get(1, 0, 0).toInt(), "Value at [1,0,0] should be 1")
    }

    @Test
    fun test4DTensorCreation() {
        // Test 4D tensor creation (2x2x2x2 with 16 Int4 values)
        val shape = Shape(2, 2, 2, 2)
        val values = ByteArray(16) { (it - 8).toByte() } // [-8, -7, ..., 7]
        val bytes = createInt4Bytes(values)
        
        val tensor = Int4TensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(16, tensor.shape.volume, "Volume should be 16")
        
        // Verify first and last values
        assertEquals(-8, tensor.get(0, 0, 0, 0).toInt(), "First value should be -8")
        assertEquals(7, tensor.get(1, 1, 1, 1).toInt(), "Last value should be 7")
    }

    @Test
    fun testInt4ValueRange() {
        // Test all valid Int4 values [-8 to 7]
        val shape = Shape(16)
        val values = ByteArray(16) { (it - 8).toByte() }
        val bytes = createInt4Bytes(values)
        
        val tensor = Int4TensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Tensor with full range should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        
        // Verify all values
        for (i in 0..15) {
            val expectedValue = (i - 8).toByte()
            assertEquals(expectedValue, tensor.get(i), "Value at index $i should be $expectedValue")
        }
    }

    @Test
    fun testOddNumberOfValues() {
        // Test odd number of values (7 values packed into 4 bytes)
        val shape = Shape(7)
        val values = byteArrayOf(-3, -2, -1, 0, 1, 2, 3)
        val bytes = createInt4Bytes(values)
        
        val tensor = Int4TensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Tensor with odd number of values should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(7, tensor.shape.volume, "Volume should be 7")
        
        // Verify all values
        for (i in values.indices) {
            assertEquals(values[i], tensor.get(i), "Value at index $i should match")
        }
    }

    @Test
    fun testErrorConditionWrongByteArraySize() {
        // Test error when byte array size doesn't match expected packed size
        val shape = Shape(5) // Needs 3 bytes (ceil(5/2))
        val wrongSizeBytes = ByteArray(2) // Too small
        
        assertFailsWith<IllegalArgumentException>("Should fail with wrong byte array size") {
            Int4TensorFactory.fromByteArray(shape, wrongSizeBytes)
        }
    }

    @Test
    fun testErrorConditionInvalidShape() {
        // Test error conditions with invalid shapes
        val bytes = createInt4Bytes(byteArrayOf(1, 2, 3, 4))
        
        assertFailsWith<IllegalArgumentException>("Should fail with 0D shape") {
            Int4TensorFactory.fromByteArray(Shape(), bytes)
        }
        
        assertFailsWith<IllegalArgumentException>("Should fail with 5D shape") {
            Int4TensorFactory.fromByteArray(Shape(2, 2, 2, 2, 2), bytes)
        }
    }

    @Test
    fun testErrorConditionEmptyByteArray() {
        // Test error when byte array is empty but shape requires data
        val shape = Shape(1)
        val emptyBytes = ByteArray(0)
        
        assertFailsWith<IllegalArgumentException>("Should fail with empty byte array") {
            Int4TensorFactory.fromByteArray(shape, emptyBytes)
        }
    }

    @Test
    fun testMinimalTensor() {
        // Test minimal 1-element tensor
        val shape = Shape(1)
        val values = byteArrayOf(5)
        val bytes = createInt4Bytes(values)
        
        val tensor = Int4TensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Minimal tensor should be created")
        assertEquals(shape, tensor.shape, "Shape should match")
        assertEquals(1, tensor.shape.volume, "Volume should be 1")
        assertEquals(5.toByte(), tensor.get(0), "Single value should be 5")
    }

    @Test
    fun testSingleDimensionTensors() {
        // Test various single dimension sizes
        val sizes = intArrayOf(2, 8, 15, 32)
        
        for (size in sizes) {
            val shape = Shape(size)
            val values = ByteArray(size) { (it % 16 - 8).toByte() }
            val bytes = createInt4Bytes(values)
            
            val tensor = Int4TensorFactory.fromByteArray(shape, bytes)
            
            assertNotNull(tensor, "Tensor of size $size should be created")
            assertEquals(shape, tensor.shape, "Shape should match for size $size")
            assertEquals(size, tensor.shape.volume, "Volume should match for size $size")
            
            // Spot check first and last values
            assertEquals(values[0], tensor.get(0), "First value should match for size $size")
            if (size > 1) {
                assertEquals(values[size - 1], tensor.get(size - 1), "Last value should match for size $size")
            }
        }
    }

    @Test
    fun testValueClamping() {
        // Test that values outside [-8, 7] range are properly handled during packing
        val shape = Shape(4)
        val extremeValues = byteArrayOf(-20, 50, -128, 127)  // Outside Int4 range
        val bytes = createInt4Bytes(extremeValues)
        
        val tensor = Int4TensorFactory.fromByteArray(shape, bytes)
        
        assertNotNull(tensor, "Tensor with clamped values should be created")
        
        // Verify values are clamped to Int4 range
        assertEquals(-8, tensor.get(0).toInt(), "Value should be clamped to -8")
        assertEquals(7, tensor.get(1).toInt(), "Value should be clamped to 7")
        assertEquals(-8, tensor.get(2).toInt(), "Value should be clamped to -8")
        assertEquals(7, tensor.get(3).toInt(), "Value should be clamped to 7")
    }

    @Test
    fun testPackingUnpackingConsistency() {
        // Test that packing and unpacking produces consistent results
        val shape = Shape(10)
        val originalValues = byteArrayOf(-8, -5, -2, -1, 0, 1, 2, 3, 6, 7)
        val bytes = createInt4Bytes(originalValues)
        
        val tensor = Int4TensorFactory.fromByteArray(shape, bytes)
        
        // Verify all values match original
        for (i in originalValues.indices) {
            assertEquals(originalValues[i], tensor.get(i), 
                "Packed/unpacked value at index $i should match original")
        }
    }
}