package sk.ainet.core.tensor.multiplication

import sk.ainet.core.tensor.*
import sk.ainet.core.tensor.backend.CpuBackend
import sk.ainet.core.tensor.backend.CpuTensorFP32
import kotlin.test.*

class EdgeCasesTest {
    
    private val backend = CpuBackend()
    
    @Test
    fun testMinimalDimensionMultiplication() {
        // Test 1x1 matrix multiplication
        val shape1 = Shape(1, 1)
        val shape2 = Shape(1, 1)
        val data1 = floatArrayOf(5f)
        val data2 = floatArrayOf(3f)
        
        val tensor1 = CpuTensorFP32.fromArray(shape1, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape2, data2)
        
        val result = backend.matmul(tensor1, tensor2)
        
        assertEquals(Shape(1, 1), result.shape)
        assertEquals(15f, result[0, 0]) // 5 * 3 = 15
    }
    
    @Test
    fun testSingleElementVectorMultiplication() {
        // Test 1x1 * 1x1 as vector operations
        val vector1Shape = Shape(1)
        val vector2Shape = Shape(1)
        val data1 = floatArrayOf(7f)
        val data2 = floatArrayOf(2f)
        
        val vector1 = CpuTensorFP32.fromArray(vector1Shape, data1)
        val vector2 = CpuTensorFP32.fromArray(vector2Shape, data2)
        
        // Element-wise multiplication
        val elementResult = with(backend) { vector1 * vector2 }
        assertEquals(Shape(1), elementResult.shape)
        assertEquals(14f, elementResult[0]) // 7 * 2 = 14
        
        // Dot product
        val dotResult = backend.dot(vector1, vector2)
        assertEquals(14.0, dotResult, 1e-6) // 7 * 2 = 14
    }
    
    @Test
    fun testVeryLargeNumbers() {
        val shape = Shape(2, 2)
        val data1 = floatArrayOf(1e6f, 2e6f, 3e6f, 4e6f)
        val data2 = floatArrayOf(5e6f, 6e6f, 7e6f, 8e6f)
        
        val tensor1 = CpuTensorFP32.fromArray(shape, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape, data2)
        
        val result = with(backend) { tensor1 * tensor2 }
        
        assertEquals(shape, result.shape)
        assertEquals(5e12f, result[0, 0], 1e6f) // 1e6 * 5e6 = 5e12
        assertEquals(12e12f, result[0, 1], 1e6f) // 2e6 * 6e6 = 12e12
        assertEquals(21e12f, result[1, 0], 1e6f) // 3e6 * 7e6 = 21e12
        assertEquals(32e12f, result[1, 1], 1e6f) // 4e6 * 8e6 = 32e12
    }
    
    @Test
    fun testVerySmallNumbers() {
        val shape = Shape(2, 2)
        val data1 = floatArrayOf(1e-6f, 2e-6f, 3e-6f, 4e-6f)
        val data2 = floatArrayOf(5e-6f, 6e-6f, 7e-6f, 8e-6f)
        
        val tensor1 = CpuTensorFP32.fromArray(shape, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape, data2)
        
        val result = with(backend) { tensor1 * tensor2 }
        
        assertEquals(shape, result.shape)
        assertEquals(5e-12f, result[0, 0], 1e-15f) // 1e-6 * 5e-6 = 5e-12
        assertEquals(12e-12f, result[0, 1], 1e-15f) // 2e-6 * 6e-6 = 12e-12
        assertEquals(21e-12f, result[1, 0], 1e-15f) // 3e-6 * 7e-6 = 21e-12
        assertEquals(32e-12f, result[1, 1], 1e-15f) // 4e-6 * 8e-6 = 32e-12
    }
    
    @Test
    fun testInfinityHandling() {
        val shape = Shape(2, 2)
        val data1 = floatArrayOf(Float.POSITIVE_INFINITY, 1f, 2f, 3f)
        val data2 = floatArrayOf(2f, Float.NEGATIVE_INFINITY, 4f, 5f)
        
        val tensor1 = CpuTensorFP32.fromArray(shape, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape, data2)
        
        val result = with(backend) { tensor1 * tensor2 }
        
        assertEquals(shape, result.shape)
        assertEquals(Float.POSITIVE_INFINITY, result[0, 0]) // +∞ * 2 = +∞
        assertEquals(Float.NEGATIVE_INFINITY, result[0, 1]) // 1 * -∞ = -∞
        assertEquals(8f, result[1, 0]) // 2 * 4 = 8
        assertEquals(15f, result[1, 1]) // 3 * 5 = 15
    }
    
    @Test
    fun testNaNHandling() {
        val shape = Shape(2, 2)
        val data1 = floatArrayOf(Float.NaN, 1f, 2f, 3f)
        val data2 = floatArrayOf(2f, Float.NaN, 4f, 5f)
        
        val tensor1 = CpuTensorFP32.fromArray(shape, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape, data2)
        
        val result = with(backend) { tensor1 * tensor2 }
        
        assertEquals(shape, result.shape)
        assertTrue(result[0, 0].isNaN()) // NaN * 2 = NaN
        assertTrue(result[0, 1].isNaN()) // 1 * NaN = NaN
        assertEquals(8f, result[1, 0]) // 2 * 4 = 8
        assertEquals(15f, result[1, 1]) // 3 * 5 = 15
    }
    
    @Test
    fun testZeroDivisionInElementwise() {
        val shape = Shape(2, 2)
        val data1 = floatArrayOf(1f, 2f, 3f, 4f)
        val data2 = floatArrayOf(0f, 2f, 0f, 4f)
        
        val tensor1 = CpuTensorFP32.fromArray(shape, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape, data2)
        
        val result = with(backend) { tensor1 / tensor2 }
        
        assertEquals(shape, result.shape)
        assertEquals(Float.POSITIVE_INFINITY, result[0, 0]) // 1 / 0 = +∞
        assertEquals(1f, result[0, 1]) // 2 / 2 = 1
        assertEquals(Float.POSITIVE_INFINITY, result[1, 0]) // 3 / 0 = +∞
        assertEquals(1f, result[1, 1]) // 4 / 4 = 1
    }
    
    @Test
    fun testNegativeZeroDivision() {
        val shape = Shape(2, 1)
        val data1 = floatArrayOf(-1f, 1f)
        val data2 = floatArrayOf(0f, 0f)
        
        val tensor1 = CpuTensorFP32.fromArray(shape, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape, data2)
        
        val result = with(backend) { tensor1 / tensor2 }
        
        assertEquals(shape, result.shape)
        assertEquals(Float.NEGATIVE_INFINITY, result[0, 0]) // -1 / 0 = -∞
        assertEquals(Float.POSITIVE_INFINITY, result[1, 0]) // 1 / 0 = +∞
    }
    
    @Test
    fun testOverflowInMatrixMultiplication() {
        // Test potential overflow in matrix multiplication
        val shape1 = Shape(2, 2)
        val shape2 = Shape(2, 2)
        val data1 = floatArrayOf(Float.MAX_VALUE / 2, 0f, 0f, Float.MAX_VALUE / 2)
        val data2 = floatArrayOf(3f, 0f, 0f, 3f)
        
        val tensor1 = CpuTensorFP32.fromArray(shape1, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape2, data2)
        
        val result = backend.matmul(tensor1, tensor2)
        
        assertEquals(shape1, result.shape)
        // Should overflow to infinity
        assertEquals(Float.POSITIVE_INFINITY, result[0, 0])
        assertEquals(0f, result[0, 1])
        assertEquals(0f, result[1, 0])
        assertEquals(Float.POSITIVE_INFINITY, result[1, 1])
    }
    
    @Test
    fun testUnderflowInMultiplication() {
        val shape = Shape(2, 2)
        val data1 = floatArrayOf(Float.MIN_VALUE, Float.MIN_VALUE, Float.MIN_VALUE, Float.MIN_VALUE)
        val data2 = floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f)
        
        val tensor1 = CpuTensorFP32.fromArray(shape, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape, data2)
        
        val result = with(backend) { tensor1 * tensor2 }
        
        assertEquals(shape, result.shape)
        // Results should be extremely small or zero due to underflow
        assertTrue(result[0, 0] >= 0f)
        assertTrue(result[0, 1] >= 0f)
        assertTrue(result[1, 0] >= 0f)
        assertTrue(result[1, 1] >= 0f)
    }
    
    @Test
    fun testMismatchedDimensionsElementwise() {
        val shape1 = Shape(3, 2)
        val shape2 = Shape(2, 3)
        val data1 = FloatArray(6) { it.toFloat() }
        val data2 = FloatArray(6) { it.toFloat() }
        
        val tensor1 = CpuTensorFP32.fromArray(shape1, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape2, data2)
        
        assertFailsWith<IllegalArgumentException> {
            with(backend) { tensor1 * tensor2 }
        }
        
        assertFailsWith<IllegalArgumentException> {
            with(backend) { tensor1 + tensor2 }
        }
        
        assertFailsWith<IllegalArgumentException> {
            with(backend) { tensor1 - tensor2 }
        }
        
        assertFailsWith<IllegalArgumentException> {
            with(backend) { tensor1 / tensor2 }
        }
    }
    
    @Test
    fun testIncompatibleMatrixDimensions() {
        val shape1 = Shape(3, 4)
        val shape2 = Shape(5, 2) // Should be 4xN for compatibility
        val data1 = FloatArray(12) { it.toFloat() }
        val data2 = FloatArray(10) { it.toFloat() }
        
        val tensor1 = CpuTensorFP32.fromArray(shape1, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape2, data2)
        
        assertFailsWith<IllegalArgumentException> {
            backend.matmul(tensor1, tensor2)
        }
    }
    
    @Test
    fun testNon2DMatrixMultiplication() {
        val shape1 = Shape(2, 3, 4) // 3D tensor
        val shape2 = Shape(4, 5)    // 2D tensor
        val data1 = FloatArray(24) { it.toFloat() }
        val data2 = FloatArray(20) { it.toFloat() }
        
        val tensor1 = CpuTensorFP32.fromArray(shape1, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape2, data2)
        
        assertFailsWith<IllegalArgumentException> {
            backend.matmul(tensor1, tensor2)
        }
    }
    
    @Test
    fun testDotProductDimensionMismatch() {
        val shape1 = Shape(3)
        val shape2 = Shape(4)
        val data1 = FloatArray(3) { it.toFloat() }
        val data2 = FloatArray(4) { it.toFloat() }
        
        val tensor1 = CpuTensorFP32.fromArray(shape1, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape2, data2)
        
        assertFailsWith<IllegalArgumentException> {
            backend.dot(tensor1, tensor2)
        }
    }
    
    @Test
    fun testMultiplicationWithEmptyShapes() {
        // Test behavior with minimal valid shapes
        val shape1 = Shape(1)
        val shape2 = Shape(1)
        val data1 = floatArrayOf(5f)
        val data2 = floatArrayOf(3f)
        
        val tensor1 = CpuTensorFP32.fromArray(shape1, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape2, data2)
        
        val result = with(backend) { tensor1 * tensor2 }
        assertEquals(Shape(1), result.shape)
        assertEquals(15f, result[0])
    }
    
    @Test
    fun testMaxDimensionTensors() {
        // Test with 4D tensors (maximum supported)
        val shape = Shape(2, 2, 2, 2)
        val data1 = FloatArray(16) { (it + 1).toFloat() }
        val data2 = FloatArray(16) { (it % 3 + 1).toFloat() }
        
        val tensor1 = CpuTensorFP32.fromArray(shape, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape, data2)
        
        val result = with(backend) { tensor1 * tensor2 }
        
        assertEquals(shape, result.shape)
        assertEquals(1f, result[0, 0, 0, 0]) // 1 * 1 = 1
        assertEquals(4f, result[0, 0, 0, 1]) // 2 * 2 = 4
        assertEquals(9f, result[0, 0, 1, 0]) // 3 * 3 = 9
    }
    
    @Test
    fun testPrecisionLossInLargeMultiplications() {
        // Test precision with operations that might lose precision
        val shape = Shape(1000)
        val data1 = FloatArray(1000) { 1.0000001f }
        val data2 = FloatArray(1000) { 1.0000001f }
        
        val tensor1 = CpuTensorFP32.fromArray(shape, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape, data2)
        
        val result = with(backend) { tensor1 * tensor2 }
        
        assertEquals(shape, result.shape)
        // All results should be close to 1.0000002f (1.0000001 * 1.0000001)
        for (i in 0 until 1000) {
            assertTrue(result[i] > 1f)
            assertTrue(result[i] < 1.1f) // Should be very close to 1
        }
    }
}