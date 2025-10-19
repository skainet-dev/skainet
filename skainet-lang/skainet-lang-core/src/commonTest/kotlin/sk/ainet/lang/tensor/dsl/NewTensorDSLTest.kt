package sk.ainet.lang.tensor.dsl

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

/**
 * Test for the new modern DSL syntax
 */
class NewTensorDSLTest {

    @Test
    fun testTensorDSLSyntaxWithOnes() {
        val vector = with<FP32, Float>(DenseTensorDataFactory()) {
            tensor(5) { shape ->
                ones()
            }
        }
        
        assertNotNull(vector)
        assertEquals(Shape(5), vector.shape)
        // Note: We can't directly access .data property without proper TensorData implementation
        // This test verifies the DSL syntax works correctly
    }
    
    @Test
    fun testTensorDSLSyntaxWithZeros() {
        val matrix = with<FP32, Float>(DenseTensorDataFactory()) {
            tensor(3, 4) { shape ->
                zeros()
            }
        }
        
        assertNotNull(matrix)
        assertEquals(Shape(3, 4), matrix.shape)
    }
    
    @Test
    fun testTensorDSLSyntaxWithFull() {
        val vector = with<FP32, Float>(DenseTensorDataFactory()) {
            tensor(5) { shape ->
                full(2.5f)
            }
        }
        
        assertNotNull(vector)
        assertEquals(Shape(5), vector.shape)
    }
    
    @Test
    fun testTensorDSLSyntaxWithRandn() {
        val matrix = with<FP32, Float>(DenseTensorDataFactory()) {
            tensor(2, 3) { shape ->
                randn(mean = 0.0f, std = 1.0f)
            }
        }
        
        assertNotNull(matrix)
        assertEquals(Shape(2, 3), matrix.shape)
    }
    
    @Test
    fun testTensorDSLSyntaxWithUniform() {
        val vector = with<FP32, Float>(DenseTensorDataFactory()) {
            tensor(10) { shape ->
                uniform(min = -1.0f, max = 1.0f)
            }
        }
        
        assertNotNull(vector)
        assertEquals(Shape(10), vector.shape)
    }
    
    @Test
    fun testTensorDSLSyntaxWithCustomInit() {
        val vector = with<FP32, Float>(DenseTensorDataFactory()) {
            tensor(5) { shape ->
                init { indices -> indices[0].toFloat() }
            }
        }
        
        assertNotNull(vector)
        assertEquals(Shape(5), vector.shape)
    }
}