package sk.ainet.core.tensor.multiplication

import sk.ai.net.core.tensor.*
import sk.ainet.core.tensor.backend.CpuBackend
import sk.ainet.core.tensor.backend.CpuTensorFP32
import kotlin.test.*

class ElementwiseMultiplicationTest {
    
    private val backend = CpuBackend()
    
    @Test
    fun testBasicElementwiseMultiplication() {
        val shape = Shape(2, 3)
        val data1 = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        val data2 = floatArrayOf(2f, 3f, 4f, 5f, 6f, 7f)
        val tensor1 = CpuTensorFP32.fromArray(shape, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape, data2)
        
        val result = with(backend) { tensor1 * tensor2 }
        
        assertEquals(shape, result.shape)
        assertEquals(2f, result[0, 0])   // 1 * 2
        assertEquals(6f, result[0, 1])   // 2 * 3
        assertEquals(12f, result[0, 2])  // 3 * 4
        assertEquals(20f, result[1, 0])  // 4 * 5
        assertEquals(30f, result[1, 1])  // 5 * 6
        assertEquals(42f, result[1, 2])  // 6 * 7
    }
    
    @Test
    fun testElementwiseMultiplication1D() {
        val shape = Shape(5)
        val data1 = floatArrayOf(1f, 2f, 3f, 4f, 5f)
        val data2 = floatArrayOf(2f, 3f, 4f, 5f, 6f)
        val tensor1 = CpuTensorFP32.fromArray(shape, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape, data2)
        
        val result = with(backend) { tensor1 * tensor2 }
        
        assertEquals(shape, result.shape)
        assertEquals(2f, result[0])   // 1 * 2
        assertEquals(6f, result[1])   // 2 * 3
        assertEquals(12f, result[2])  // 3 * 4
        assertEquals(20f, result[3])  // 4 * 5
        assertEquals(30f, result[4])  // 5 * 6
    }
    
    @Test
    fun testElementwiseMultiplication3D() {
        val shape = Shape(2, 2, 2)
        val data1 = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)
        val data2 = floatArrayOf(2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f)
        val tensor1 = CpuTensorFP32.fromArray(shape, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape, data2)
        
        val result = with(backend) { tensor1 * tensor2 }
        
        assertEquals(shape, result.shape)
        assertEquals(2f, result[0, 0, 0])   // 1 * 2
        assertEquals(6f, result[0, 0, 1])   // 2 * 3
        assertEquals(12f, result[0, 1, 0])  // 3 * 4
        assertEquals(20f, result[0, 1, 1])  // 4 * 5
        assertEquals(30f, result[1, 0, 0])  // 5 * 6
        assertEquals(42f, result[1, 0, 1])  // 6 * 7
        assertEquals(56f, result[1, 1, 0])  // 7 * 8
        assertEquals(72f, result[1, 1, 1])  // 8 * 9
    }
    
    @Test
    fun testElementwiseMultiplication4D() {
        // NCHW format: 2 batches, 1 channel, 2x2 spatial
        val shape = Shape(2, 1, 2, 2)
        val data1 = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)
        val data2 = floatArrayOf(2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f)
        val tensor1 = CpuTensorFP32.fromArray(shape, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape, data2)
        
        val result = with(backend) { tensor1 * tensor2 }
        
        assertEquals(shape, result.shape)
        assertEquals(2f, result[0, 0, 0, 0])   // 1 * 2
        assertEquals(6f, result[0, 0, 0, 1])   // 2 * 3
        assertEquals(12f, result[0, 0, 1, 0])  // 3 * 4
        assertEquals(20f, result[0, 0, 1, 1])  // 4 * 5
        assertEquals(30f, result[1, 0, 0, 0])  // 5 * 6
        assertEquals(42f, result[1, 0, 0, 1])  // 6 * 7
        assertEquals(56f, result[1, 0, 1, 0])  // 7 * 8
        assertEquals(72f, result[1, 0, 1, 1])  // 8 * 9
    }
    
    @Test
    fun testElementwiseWithZeros() {
        val shape = Shape(3, 3)
        val data1 = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f)
        val tensor1 = CpuTensorFP32.fromArray(shape, data1)
        val tensor2 = CpuTensorFP32.zeros(shape)
        
        val result = with(backend) { tensor1 * tensor2 }
        
        assertEquals(shape, result.shape)
        for (i in 0 until 3) {
            for (j in 0 until 3) {
                assertEquals(0f, result[i, j])
            }
        }
    }
    
    @Test
    fun testElementwiseWithOnes() {
        val shape = Shape(2, 4)
        val data = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)
        val tensor1 = CpuTensorFP32.fromArray(shape, data)
        val tensor2 = CpuTensorFP32.ones(shape)
        
        val result = with(backend) { tensor1 * tensor2 }
        
        assertEquals(shape, result.shape)
        assertEquals(1f, result[0, 0])
        assertEquals(2f, result[0, 1])
        assertEquals(3f, result[0, 2])
        assertEquals(4f, result[0, 3])
        assertEquals(5f, result[1, 0])
        assertEquals(6f, result[1, 1])
        assertEquals(7f, result[1, 2])
        assertEquals(8f, result[1, 3])
    }
    
    @Test
    fun testElementwiseWithNegativeValues() {
        val shape = Shape(2, 2)
        val data1 = floatArrayOf(-1f, 2f, -3f, 4f)
        val data2 = floatArrayOf(2f, -3f, 4f, -5f)
        val tensor1 = CpuTensorFP32.fromArray(shape, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape, data2)
        
        val result = with(backend) { tensor1 * tensor2 }
        
        assertEquals(shape, result.shape)
        assertEquals(-2f, result[0, 0])  // -1 * 2 = -2
        assertEquals(-6f, result[0, 1])  // 2 * -3 = -6
        assertEquals(-12f, result[1, 0]) // -3 * 4 = -12
        assertEquals(-20f, result[1, 1]) // 4 * -5 = -20
    }
    
    @Test
    fun testElementwiseWithFractionalValues() {
        val shape = Shape(2, 2)
        val data1 = floatArrayOf(0.5f, 1.5f, 2.5f, 3.5f)
        val data2 = floatArrayOf(2f, 4f, 6f, 8f)
        val tensor1 = CpuTensorFP32.fromArray(shape, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape, data2)
        
        val result = with(backend) { tensor1 * tensor2 }
        
        assertEquals(shape, result.shape)
        assertEquals(1f, result[0, 0], 1e-6f)    // 0.5 * 2 = 1
        assertEquals(6f, result[0, 1], 1e-6f)    // 1.5 * 4 = 6
        assertEquals(15f, result[1, 0], 1e-6f)   // 2.5 * 6 = 15
        assertEquals(28f, result[1, 1], 1e-6f)   // 3.5 * 8 = 28
    }
    
    @Test
    fun testLargeElementwiseMultiplication() {
        val shape = Shape(100, 100)
        val data1 = FloatArray(10000) { (it + 1).toFloat() }
        val data2 = FloatArray(10000) { (it * 2).toFloat() }
        val tensor1 = CpuTensorFP32.fromArray(shape, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape, data2)
        
        val result = with(backend) { tensor1 * tensor2 }
        
        assertEquals(shape, result.shape)
        assertEquals(0f, result[0, 0])          // 1 * 0 = 0
        assertEquals(4f, result[0, 1])          // 2 * 2 = 4
        assertEquals(12f, result[0, 2])         // 3 * 4 = 12
        assertEquals(199980000f, result[99, 99], 1000f)   // 10000 * 19998 = 199980000
    }
    
    @Test
    fun testElementwiseShapeMismatch() {
        val shape1 = Shape(2, 3)
        val shape2 = Shape(3, 2)
        val data1 = FloatArray(6) { it.toFloat() }
        val data2 = FloatArray(6) { it.toFloat() }
        val tensor1 = CpuTensorFP32.fromArray(shape1, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape2, data2)
        
        assertFailsWith<IllegalArgumentException> {
            with(backend) { tensor1 * tensor2 }
        }
    }
}