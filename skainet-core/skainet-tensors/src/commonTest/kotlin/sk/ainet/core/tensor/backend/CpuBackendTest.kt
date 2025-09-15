package sk.ainet.core.tensor.backend

import sk.ai.net.core.tensor.Shape
import kotlin.test.*

class CpuBackendTest {
    
    private val backend = CpuBackend()
    
    @Test
    fun testTensorCreation1D() {
        val shape = Shape(5)
        val data = floatArrayOf(1f, 2f, 3f, 4f, 5f)
        val tensor = CpuTensorFP32.fromArray(shape, data)
        
        assertEquals(shape, tensor.shape)
        assertEquals(5, tensor.shape.volume)
        assertEquals(1, tensor.shape.rank)
        
        for (i in 0 until 5) {
            assertEquals(data[i], tensor[i])
        }
    }
    
    @Test
    fun testTensorCreation2D() {
        val shape = Shape(2, 3)
        val data = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        val tensor = CpuTensorFP32.fromArray(shape, data)
        
        assertEquals(shape, tensor.shape)
        assertEquals(6, tensor.shape.volume)
        assertEquals(2, tensor.shape.rank)
        
        // Test NCHW row-major layout
        assertEquals(1f, tensor[0, 0])
        assertEquals(2f, tensor[0, 1])
        assertEquals(3f, tensor[0, 2])
        assertEquals(4f, tensor[1, 0])
        assertEquals(5f, tensor[1, 1])
        assertEquals(6f, tensor[1, 2])
    }
    
    @Test
    fun testTensorCreation3D() {
        val shape = Shape(2, 2, 2)
        val data = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)
        val tensor = CpuTensorFP32.fromArray(shape, data)
        
        assertEquals(shape, tensor.shape)
        assertEquals(8, tensor.shape.volume)
        assertEquals(3, tensor.shape.rank)
        
        // Test NCHW layout for 3D tensor
        assertEquals(1f, tensor[0, 0, 0])
        assertEquals(2f, tensor[0, 0, 1])
        assertEquals(3f, tensor[0, 1, 0])
        assertEquals(4f, tensor[0, 1, 1])
        assertEquals(5f, tensor[1, 0, 0])
        assertEquals(6f, tensor[1, 0, 1])
        assertEquals(7f, tensor[1, 1, 0])
        assertEquals(8f, tensor[1, 1, 1])
    }
    
    @Test
    fun testTensorCreation4D() {
        val shape = Shape(1, 2, 2, 2) // NCHW: 1 batch, 2 channels, 2x2 spatial
        val data = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)
        val tensor = CpuTensorFP32.fromArray(shape, data)
        
        assertEquals(shape, tensor.shape)
        assertEquals(8, tensor.shape.volume)
        assertEquals(4, tensor.shape.rank)
        
        // Test NCHW layout for 4D tensor
        assertEquals(1f, tensor[0, 0, 0, 0])
        assertEquals(2f, tensor[0, 0, 0, 1])
        assertEquals(3f, tensor[0, 0, 1, 0])
        assertEquals(4f, tensor[0, 0, 1, 1])
        assertEquals(5f, tensor[0, 1, 0, 0])
        assertEquals(6f, tensor[0, 1, 0, 1])
        assertEquals(7f, tensor[0, 1, 1, 0])
        assertEquals(8f, tensor[0, 1, 1, 1])
    }
    
    @Test
    fun testZerosTensor() {
        val shape = Shape(2, 3)
        val tensor = CpuTensorFP32.zeros(shape)
        
        assertEquals(shape, tensor.shape)
        for (i in 0 until 2) {
            for (j in 0 until 3) {
                assertEquals(0f, tensor[i, j])
            }
        }
    }
    
    @Test
    fun testOnesTensor() {
        val shape = Shape(2, 3)
        val tensor = CpuTensorFP32.ones(shape)
        
        assertEquals(shape, tensor.shape)
        for (i in 0 until 2) {
            for (j in 0 until 3) {
                assertEquals(1f, tensor[i, j])
            }
        }
    }
    
    @Test
    fun testFullTensor() {
        val shape = Shape(2, 3)
        val value = 5.5f
        val tensor = CpuTensorFP32.full(shape, value)
        
        assertEquals(shape, tensor.shape)
        for (i in 0 until 2) {
            for (j in 0 until 3) {
                assertEquals(value, tensor[i, j])
            }
        }
    }
    
    @Test
    fun testTensorAddition() {
        val shape = Shape(2, 3)
        val data1 = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        val data2 = floatArrayOf(2f, 3f, 4f, 5f, 6f, 7f)
        val tensor1 = CpuTensorFP32.fromArray(shape, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape, data2)
        
        val result = with(backend) { tensor1 + tensor2 }
        
        assertEquals(shape, result.shape)
        assertEquals(3f, result[0, 0]) // 1 + 2
        assertEquals(5f, result[0, 1]) // 2 + 3
        assertEquals(7f, result[0, 2]) // 3 + 4
        assertEquals(9f, result[1, 0]) // 4 + 5
        assertEquals(11f, result[1, 1]) // 5 + 6
        assertEquals(13f, result[1, 2]) // 6 + 7
    }
    
    @Test
    fun testTensorSubtraction() {
        val shape = Shape(2, 2)
        val data1 = floatArrayOf(5f, 6f, 7f, 8f)
        val data2 = floatArrayOf(1f, 2f, 3f, 4f)
        val tensor1 = CpuTensorFP32.fromArray(shape, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape, data2)
        
        val result = with(backend) { tensor1 - tensor2 }
        
        assertEquals(shape, result.shape)
        assertEquals(4f, result[0, 0]) // 5 - 1
        assertEquals(4f, result[0, 1]) // 6 - 2
        assertEquals(4f, result[1, 0]) // 7 - 3
        assertEquals(4f, result[1, 1]) // 8 - 4
    }
    
    @Test
    fun testTensorMultiplication() {
        val shape = Shape(2, 2)
        val data1 = floatArrayOf(2f, 3f, 4f, 5f)
        val data2 = floatArrayOf(3f, 4f, 5f, 6f)
        val tensor1 = CpuTensorFP32.fromArray(shape, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape, data2)
        
        val result = with(backend) { tensor1 * tensor2 }
        
        assertEquals(shape, result.shape)
        assertEquals(6f, result[0, 0])  // 2 * 3
        assertEquals(12f, result[0, 1]) // 3 * 4
        assertEquals(20f, result[1, 0]) // 4 * 5
        assertEquals(30f, result[1, 1]) // 5 * 6
    }
    
    @Test
    fun testTensorDivision() {
        val shape = Shape(2, 2)
        val data1 = floatArrayOf(6f, 8f, 10f, 12f)
        val data2 = floatArrayOf(2f, 4f, 5f, 6f)
        val tensor1 = CpuTensorFP32.fromArray(shape, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape, data2)
        
        val result = with(backend) { tensor1 / tensor2 }
        
        assertEquals(shape, result.shape)
        assertEquals(3f, result[0, 0])  // 6 / 2
        assertEquals(2f, result[0, 1])  // 8 / 4
        assertEquals(2f, result[1, 0])  // 10 / 5
        assertEquals(2f, result[1, 1])  // 12 / 6
    }
    
    @Test
    fun testMatrixMultiplication() {
        // Test 2x3 * 3x2 = 2x2 matrix multiplication
        val shape1 = Shape(2, 3)
        val shape2 = Shape(3, 2)
        val data1 = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f) // 2x3 matrix
        val data2 = floatArrayOf(7f, 8f, 9f, 10f, 11f, 12f) // 3x2 matrix
        
        val tensor1 = CpuTensorFP32.fromArray(shape1, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape2, data2)
        
        val result = backend.matmul(tensor1, tensor2)
        
        assertEquals(Shape(2, 2), result.shape)
        // First row: [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12] = [58, 64]
        // Second row: [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12] = [139, 154]
        assertEquals(58f, result[0, 0])
        assertEquals(64f, result[0, 1])
        assertEquals(139f, result[1, 0])
        assertEquals(154f, result[1, 1])
    }
    
    @Test
    fun testDotProduct() {
        val shape = Shape(3)
        val data1 = floatArrayOf(1f, 2f, 3f)
        val data2 = floatArrayOf(4f, 5f, 6f)
        
        val tensor1 = CpuTensorFP32.fromArray(shape, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape, data2)
        
        val result = backend.dot(tensor1, tensor2)
        
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assertEquals(32.0, result, 1e-6)
    }
    
    @Test
    fun testScalarOperations() {
        val shape = Shape(2, 2)
        val data = floatArrayOf(1f, 2f, 3f, 4f)
        val tensor = CpuTensorFP32.fromArray(shape, data)
        
        // Test tensor + scalar
        val addResult = with(backend) { tensor + 5 }
        assertEquals(6f, addResult[0, 0])
        assertEquals(7f, addResult[0, 1])
        assertEquals(8f, addResult[1, 0])
        assertEquals(9f, addResult[1, 1])
        
        // Test tensor * scalar
        val mulResult = with(backend) { tensor * 2f }
        assertEquals(2f, mulResult[0, 0])
        assertEquals(4f, mulResult[0, 1])
        assertEquals(6f, mulResult[1, 0])
        assertEquals(8f, mulResult[1, 1])
        
        // Test scalar * tensor
        val scalarMulResult = with(backend) { 3.0 * tensor }
        assertEquals(3f, scalarMulResult[0, 0])
        assertEquals(6f, scalarMulResult[0, 1])
        assertEquals(9f, scalarMulResult[1, 0])
        assertEquals(12f, scalarMulResult[1, 1])
    }
    
    @Test
    fun testScale() {
        val shape = Shape(2, 2)
        val data = floatArrayOf(1f, 2f, 3f, 4f)
        val tensor = CpuTensorFP32.fromArray(shape, data)
        
        val result = backend.scale(tensor, 2.5)
        
        assertEquals(2.5f, result[0, 0])
        assertEquals(5.0f, result[0, 1])
        assertEquals(7.5f, result[1, 0])
        assertEquals(10.0f, result[1, 1])
    }
    
    @Test
    fun testInvalidDimensions() {
        // Test that 5D tensor creation fails
        val shape5D = Shape(1, 2, 3, 4, 5)
        val data = FloatArray(1 * 2 * 3 * 4 * 5) { it.toFloat() }
        
        assertFailsWith<IllegalArgumentException> {
            CpuTensorFP32.fromArray(shape5D, data)
        }
        
        // Test that 0D tensor creation fails
        val shape0D = Shape()
        val data0D = floatArrayOf(1f)
        
        assertFailsWith<IllegalArgumentException> {
            CpuTensorFP32.fromArray(shape0D, data0D)
        }
    }
    
    @Test
    fun testDataSizeMismatch() {
        val shape = Shape(2, 3)
        val wrongSizeData = floatArrayOf(1f, 2f, 3f, 4f) // Should be 6 elements, not 4
        
        assertFailsWith<IllegalArgumentException> {
            CpuTensorFP32.fromArray(shape, wrongSizeData)
        }
    }
    
    @Test
    fun testMatrixMultiplicationInvalidDimensions() {
        val shape1 = Shape(2, 3)
        val shape2 = Shape(4, 2) // Should be 3x2 for valid multiplication
        val data1 = FloatArray(6) { it.toFloat() }
        val data2 = FloatArray(8) { it.toFloat() }
        
        val tensor1 = CpuTensorFP32.fromArray(shape1, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape2, data2)
        
        assertFailsWith<IllegalArgumentException> {
            backend.matmul(tensor1, tensor2)
        }
    }
    
    @Test
    fun testTensorOperationsShapeMismatch() {
        val shape1 = Shape(2, 3)
        val shape2 = Shape(3, 2)
        val data1 = FloatArray(6) { it.toFloat() }
        val data2 = FloatArray(6) { it.toFloat() }
        
        val tensor1 = CpuTensorFP32.fromArray(shape1, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape2, data2)
        
        assertFailsWith<IllegalArgumentException> {
            with(backend) { tensor1 + tensor2 }
        }
        
        assertFailsWith<IllegalArgumentException> {
            with(backend) { tensor1 - tensor2 }
        }
        
        assertFailsWith<IllegalArgumentException> {
            with(backend) { tensor1 * tensor2 }
        }
        
        assertFailsWith<IllegalArgumentException> {
            with(backend) { tensor1 / tensor2 }
        }
    }
    
    @Test
    fun testNCHWLayoutPreservation() {
        // Test that NCHW layout is preserved through operations
        val shape = Shape(2, 3, 2, 2) // 2 batches, 3 channels, 2x2 spatial
        val data = FloatArray(24) { it.toFloat() + 1f }
        val tensor = CpuTensorFP32.fromArray(shape, data)
        
        // Test that indexing follows NCHW order
        assertEquals(1f, tensor[0, 0, 0, 0])   // First element
        assertEquals(2f, tensor[0, 0, 0, 1])   // Next in width
        assertEquals(3f, tensor[0, 0, 1, 0])   // Next in height
        assertEquals(4f, tensor[0, 0, 1, 1])   // Complete first channel
        assertEquals(5f, tensor[0, 1, 0, 0])   // First element of second channel
        assertEquals(13f, tensor[1, 0, 0, 0])  // First element of second batch
        
        // Test that operations preserve layout
        val doubled = with(backend) { tensor * 2f }
        assertEquals(2f, doubled[0, 0, 0, 0])
        assertEquals(4f, doubled[0, 0, 0, 1])
        assertEquals(6f, doubled[0, 0, 1, 0])
        assertEquals(10f, doubled[0, 1, 0, 0])
        assertEquals(26f, doubled[1, 0, 0, 0])
    }
}