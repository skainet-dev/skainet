package sk.ainet.core.tensor

import sk.ainet.core.tensor.backend.CpuTensorFP32
import sk.ainet.core.tensor.backend.CpuTensorInt32
import kotlin.test.*

/**
 * Tests for creating 2D tensors from nested list structures.
 * Tests the new fromNestedList factory methods that work similar to numpy.array([[...], [...]])
 */
class NestedListTensorFactoryTest {
    
    @Test
    fun testInt32FromNestedListBasic() {
        // Test basic 2D tensor creation similar to np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
        val data = listOf(
            listOf(2, 1, 4, 3),
            listOf(1, 2, 3, 4),
            listOf(4, 3, 2, 1)
        )
        
        val tensor = CpuTensorInt32.fromNestedList(data)
        
        assertEquals(Shape(3, 4), tensor.shape, "Tensor should have shape (3, 4)")
        
        // Verify data is stored in row-major order
        val expectedData = intArrayOf(2, 1, 4, 3, 1, 2, 3, 4, 4, 3, 2, 1)
        assertContentEquals(expectedData, tensor.data, "Tensor data should match expected row-major order")
    }
    
    @Test
    fun testFP32FromNestedListBasic() {
        // Test basic 2D float tensor creation
        val data = listOf(
            listOf(2.5f, 1.0f, 4.2f),
            listOf(1.5f, 2.8f, 3.1f)
        )
        
        val tensor = CpuTensorFP32.fromNestedList(data)
        
        assertEquals(Shape(2, 3), tensor.shape, "Tensor should have shape (2, 3)")
        
        // Verify data is stored in row-major order
        val expectedData = floatArrayOf(2.5f, 1.0f, 4.2f, 1.5f, 2.8f, 3.1f)
        assertContentEquals(expectedData, tensor.data, "Tensor data should match expected row-major order")
    }
    
    @Test
    fun testSingleRowTensor() {
        // Test creating a tensor from a single row
        val data = listOf(listOf(1, 2, 3, 4, 5))
        val tensor = CpuTensorInt32.fromNestedList(data)
        
        assertEquals(Shape(1, 5), tensor.shape, "Single row tensor should have shape (1, 5)")
        assertContentEquals(intArrayOf(1, 2, 3, 4, 5), tensor.data)
    }
    
    @Test
    fun testSingleColumnTensor() {
        // Test creating a tensor from a single column
        val data = listOf(
            listOf(1),
            listOf(2),
            listOf(3)
        )
        val tensor = CpuTensorInt32.fromNestedList(data)
        
        assertEquals(Shape(3, 1), tensor.shape, "Single column tensor should have shape (3, 1)")
        assertContentEquals(intArrayOf(1, 2, 3), tensor.data)
    }
    
    @Test
    fun testSingleElementTensor() {
        // Test creating a tensor from a single element
        val data = listOf(listOf(42))
        val tensor = CpuTensorInt32.fromNestedList(data)
        
        assertEquals(Shape(1, 1), tensor.shape, "Single element tensor should have shape (1, 1)")
        assertContentEquals(intArrayOf(42), tensor.data)
    }
    
    @Test
    fun testLargerTensor() {
        // Test with a larger tensor to ensure scaling works
        val data = (1..5).map { row ->
            (1..6).map { col -> row * 10 + col }.toList()
        }.toList()
        
        val tensor = CpuTensorInt32.fromNestedList(data)
        
        assertEquals(Shape(5, 6), tensor.shape, "Large tensor should have shape (5, 6)")
        assertEquals(30, tensor.data.size, "Tensor should have 30 elements")
        
        // Verify first few elements
        assertEquals(11, tensor.data[0])  // Row 1, Col 1
        assertEquals(16, tensor.data[5])  // Row 1, Col 6
        assertEquals(21, tensor.data[6])  // Row 2, Col 1
    }
    
    @Test
    fun testEmptyDataThrowsException() {
        // Test that empty data throws an exception
        val emptyData = emptyList<List<Int>>()
        
        assertFailsWith<IllegalArgumentException> {
            CpuTensorInt32.fromNestedList(emptyData)
        }
    }
    
    @Test
    fun testEmptyRowThrowsException() {
        // Test that empty rows throw an exception
        val dataWithEmptyRow = listOf(emptyList<Int>())
        
        assertFailsWith<IllegalArgumentException> {
            CpuTensorInt32.fromNestedList(dataWithEmptyRow)
        }
    }
    
    @Test
    fun testInconsistentRowSizesThrowsException() {
        // Test that inconsistent row sizes throw an exception
        val inconsistentData = listOf(
            listOf(1, 2, 3),
            listOf(4, 5),  // Different size
            listOf(6, 7, 8)
        )
        
        val exception = assertFailsWith<IllegalArgumentException> {
            CpuTensorInt32.fromNestedList(inconsistentData)
        }
        
        assertTrue(exception.message?.contains("All rows must have the same number of columns") == true,
                  "Exception should mention row size mismatch")
    }
    
    @Test
    fun testInconsistentRowSizesDetailedMessage() {
        // Test that the exception message provides detailed information
        val inconsistentData = listOf(
            listOf(1, 2, 3, 4),  // 4 columns
            listOf(5, 6),        // 2 columns
            listOf(7, 8, 9, 10)  // 4 columns
        )
        
        val exception = assertFailsWith<IllegalArgumentException> {
            CpuTensorInt32.fromNestedList(inconsistentData)
        }
        
        val message = exception.message
        assertTrue(message?.contains("Row 0 has 4 columns") == true, "Should mention first row column count")
        assertTrue(message?.contains("row 1 has 2 columns") == true, "Should mention problematic row column count")
    }
    
    @Test
    fun testFloatTensorValidation() {
        // Test validation also works for float tensors
        val inconsistentFloatData = listOf(
            listOf(1.0f, 2.0f, 3.0f),
            listOf(4.0f, 5.0f)  // Different size
        )
        
        assertFailsWith<IllegalArgumentException> {
            CpuTensorFP32.fromNestedList(inconsistentFloatData)
        }
    }
    
    @Test
    fun testNegativeValues() {
        // Test that negative values work correctly
        val dataWithNegatives = listOf(
            listOf(-1, -2, 3),
            listOf(4, -5, -6)
        )
        
        val tensor = CpuTensorInt32.fromNestedList(dataWithNegatives)
        
        assertEquals(Shape(2, 3), tensor.shape)
        assertContentEquals(intArrayOf(-1, -2, 3, 4, -5, -6), tensor.data)
    }
    
    @Test
    fun testFloatPrecision() {
        // Test that float precision is preserved
        val precisionData = listOf(
            listOf(1.123456f, 2.987654f),
            listOf(3.14159f, 2.71828f)
        )
        
        val tensor = CpuTensorFP32.fromNestedList(precisionData)
        
        assertEquals(Shape(2, 2), tensor.shape)
        assertEquals(1.123456f, tensor.data[0], 0.000001f, "First element precision should be preserved")
        assertEquals(2.987654f, tensor.data[1], 0.000001f, "Second element precision should be preserved")
        assertEquals(3.14159f, tensor.data[2], 0.000001f, "Third element precision should be preserved")
        assertEquals(2.71828f, tensor.data[3], 0.000001f, "Fourth element precision should be preserved")
    }
}