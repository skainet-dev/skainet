package sk.ai.net.core.tensor

import sk.ai.net.core.tensor.backend.*
import kotlin.test.*

class PrintingTest {
    
    @Test
    fun testScalarPrinting() {
        // Test scalar (1D tensor with single element)
        val scalar = CpuTensorFP32.fromArray(Shape(1), floatArrayOf(42.5f))
        
        val scalarOutput = scalar.printScalar()
        assertEquals("42.5", scalarOutput)
        
        val generalOutput = scalar.print()
        assertEquals("42.5", generalOutput)
    }
    
    @Test
    fun testVectorPrinting() {
        // Test vector (1D tensor)
        val vector = CpuTensorFP32.fromArray(Shape(4), floatArrayOf(1.0f, 2.5f, -3.0f, 4.7f))
        
        val vectorOutput = vector.printVector()
        assertEquals("[1.0, 2.5, -3.0, 4.7]", vectorOutput)
        
        val generalOutput = vector.print()
        assertEquals("[1.0, 2.5, -3.0, 4.7]", generalOutput)
    }
    
    @Test
    fun testMatrixPrinting() {
        // Test matrix (2D tensor)
        val matrix = CpuTensorFP32.fromArray(
            Shape(2, 3), 
            floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)
        )
        
        val matrixOutput = matrix.printMatrix()
        val expectedMatrix = """[
  [1.0, 2.0, 3.0],
  [4.0, 5.0, 6.0]
]"""
        assertEquals(expectedMatrix, matrixOutput)
        
        val generalOutput = matrix.print()
        assertEquals(expectedMatrix, generalOutput)
    }
    
    @Test
    fun testSingleElementMatrixPrinting() {
        // Test 1x1 matrix
        val singleMatrix = CpuTensorFP32.fromArray(Shape(1, 1), floatArrayOf(7.5f))
        
        val matrixOutput = singleMatrix.printMatrix()
        val expectedMatrix = """[
  [7.5]
]"""
        assertEquals(expectedMatrix, matrixOutput)
        
        val generalOutput = singleMatrix.print()
        assertEquals(expectedMatrix, generalOutput)
    }
    
    @Test
    fun testLargerMatrixPrinting() {
        // Test larger matrix
        val largerMatrix = CpuTensorFP32.fromArray(
            Shape(3, 2), 
            floatArrayOf(1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f)
        )
        
        val matrixOutput = largerMatrix.printMatrix()
        val expectedMatrix = """[
  [1.1, 2.2],
  [3.3, 4.4],
  [5.5, 6.6]
]"""
        assertEquals(expectedMatrix, matrixOutput)
    }
    
    @Test
    fun testWrongRankForScalar() {
        // Test error handling for wrong tensor rank in printScalar
        val vector = CpuTensorFP32.fromArray(Shape(3), floatArrayOf(1.0f, 2.0f, 3.0f))
        
        assertFailsWith<IllegalArgumentException> {
            vector.printScalar()
        }
    }
    
    @Test
    fun testWrongRankForVector() {
        // Test error handling for wrong tensor rank in printVector
        val matrix = CpuTensorFP32.fromArray(Shape(2, 2), floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f))
        
        assertFailsWith<IllegalArgumentException> {
            matrix.printVector()
        }
    }
    
    @Test
    fun testWrongRankForMatrix() {
        // Test error handling for wrong tensor rank in printMatrix
        val vector = CpuTensorFP32.fromArray(Shape(3), floatArrayOf(1.0f, 2.0f, 3.0f))
        
        assertFailsWith<IllegalArgumentException> {
            vector.printMatrix()
        }
    }
    
    @Test
    fun testHigherDimensionTensorPrinting() {
        // Test that higher-dimensional tensors show appropriate message
        val tensor3D = CpuTensorFP32.fromArray(
            Shape(2, 2, 2), 
            FloatArray(8) { it.toFloat() }
        )
        
        val output = tensor3D.print()
        assertTrue(output.contains("printing not supported for tensors with rank > 2"))
        assertTrue(output.contains("rank=3"))
    }
}