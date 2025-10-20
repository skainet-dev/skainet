package sk.ainet.lang.tensor.ops

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class VoidTensorOpsTest {
    
    private val dataFactory = DenseTensorDataFactory()
    private val ops = VoidTensorOps<Float>()
    
    // Helper function to create test tensors
    private fun createTensor(shape: Shape): VoidOpsTensor<FP32, Float> {
        val data = dataFactory.zeros<FP32, Float>(shape, FP32::class)
        return VoidOpsTensor(data, FP32::class)
    }
    
    private fun createIntTensor(shape: Shape): VoidOpsTensor<Int32, Int> {
        val intOps = VoidTensorOps<Int>()
        val data = DenseTensorDataFactory().zeros<Int32, Int>(shape, Int32::class)
        return VoidOpsTensor(data, Int32::class)
    }
    
    @Test
    fun testAdd_SameShape_Success() {
        val a = createTensor(Shape(2, 3))
        val b = createTensor(Shape(2, 3))
        
        val result = ops.add(a, b)
        
        assertEquals(Shape(2, 3), result.shape)
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testAdd_DifferentShape_ThrowsException() {
        val a = createTensor(Shape(2, 3))
        val b = createTensor(Shape(3, 2))
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.add(a, b)
        }
        assertTrue(exception.message?.contains("Shape mismatch for addition") == true)
    }
    
    @Test
    fun testSubtract_SameShape_Success() {
        val a = createTensor(Shape(4, 1))
        val b = createTensor(Shape(4, 1))
        
        val result = ops.subtract(a, b)
        
        assertEquals(Shape(4, 1), result.shape)
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testMultiply_ScalarTensors_Success() {
        val a = createTensor(Shape(1))
        val b = createTensor(Shape(1))
        
        val result = ops.multiply(a, b)
        
        assertEquals(Shape(1), result.shape)
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testDivide_DifferentShape_ThrowsException() {
        val a = createTensor(Shape(2, 3))
        val b = createTensor(Shape(3, 2))
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.divide(a, b)
        }
        assertTrue(exception.message?.contains("Shape mismatch for division") == true)
    }
    
    @Test
    fun testMatmul_2D_Success() {
        val a = createTensor(Shape(3, 4))  // 3x4
        val b = createTensor(Shape(4, 5))  // 4x5
        
        val result = ops.matmul(a, b)
        
        assertEquals(Shape(3, 5), result.shape)  // 3x5
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testMatmul_BatchedMatrices_Success() {
        val a = createTensor(Shape(2, 3, 4))  // batch=2, 3x4 matrices
        val b = createTensor(Shape(2, 4, 5))  // batch=2, 4x5 matrices
        
        val result = ops.matmul(a, b)
        
        assertEquals(Shape(2, 3, 5), result.shape)  // batch=2, 3x5 matrices
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testMatmul_IncompatibleInnerDimensions_ThrowsException() {
        val a = createTensor(Shape(3, 4))  // 3x4
        val b = createTensor(Shape(5, 2))  // 5x2 (incompatible: 4 != 5)
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.matmul(a, b)
        }
        assertTrue(exception.message?.contains("inner dimensions must match") == true)
    }
    
    @Test
    fun testMatmul_1DTensor_ThrowsException() {
        val a = createTensor(Shape(4))
        val b = createTensor(Shape(4))
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.matmul(a, b)
        }
        assertTrue(exception.message?.contains("at least 2 dimensions") == true)
    }
    
    @Test
    fun testTranspose_2D_Success() {
        val tensor = createTensor(Shape(3, 4))
        
        val result = ops.transpose(tensor)
        
        assertEquals(Shape(4, 3), result.shape)
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testTranspose_3D_Success() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val result = ops.transpose(tensor)
        
        assertEquals(Shape(2, 4, 3), result.shape)  // Last two dimensions swapped
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testTranspose_1D_ThrowsException() {
        val tensor = createTensor(Shape(5))
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.transpose(tensor)
        }
        assertTrue(exception.message?.contains("at least 2 dimensions") == true)
    }
    
    @Test
    fun testReshape_SameVolume_Success() {
        val tensor = createTensor(Shape(2, 3, 4))  // Volume = 24
        val newShape = Shape(6, 4)  // Volume = 24
        
        val result = ops.reshape(tensor, newShape)
        
        assertEquals(newShape, result.shape)
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testReshape_DifferentVolume_ThrowsException() {
        val tensor = createTensor(Shape(2, 3))  // Volume = 6
        val newShape = Shape(3, 3)  // Volume = 9
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.reshape(tensor, newShape)
        }
        assertTrue(exception.message?.contains("volume mismatch") == true)
    }
    
    @Test
    fun testFlatten_MiddleDimensions_Success() {
        val tensor = createTensor(Shape(2, 3, 4, 5))  // 4D tensor
        
        val result = ops.flatten(tensor, 1, 2)  // Flatten dimensions 1 and 2
        
        assertEquals(Shape(2, 12, 5), result.shape)  // 2, (3*4), 5
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testFlatten_AllDimensions_Success() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val result = ops.flatten(tensor, 0, -1)  // Flatten all dimensions (negative indexing)
        
        assertEquals(Shape(24), result.shape)  // 2*3*4 = 24
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testFlatten_InvalidDimensions_ThrowsException() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.flatten(tensor, 5, 6)  // Out of bounds
        }
        assertTrue(exception.message?.contains("out of bounds") == true)
    }
    
    @Test
    fun testFlatten_StartGreaterThanEnd_ThrowsException() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.flatten(tensor, 2, 1)  // start > end
        }
        assertTrue(exception.message?.contains("must be <=") == true)
    }
    
    @Test
    fun testActivationFunctions_PreserveShape() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val reluResult = ops.relu(tensor)
        val sigmoidResult = ops.sigmoid(tensor)
        val siluResult = ops.silu(tensor)
        val geluResult = ops.gelu(tensor)
        
        assertEquals(tensor.shape, reluResult.shape)
        assertEquals(tensor.shape, sigmoidResult.shape)
        assertEquals(tensor.shape, siluResult.shape)
        assertEquals(tensor.shape, geluResult.shape)
        assertEquals(FP32::class, reluResult.dtype)
        assertEquals(FP32::class, sigmoidResult.dtype)
        assertEquals(FP32::class, siluResult.dtype)
        assertEquals(FP32::class, geluResult.dtype)
    }
    
    @Test
    fun testSoftmax_ValidDimension_Success() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val result = ops.softmax(tensor, 1)  // Apply softmax along dimension 1
        
        assertEquals(tensor.shape, result.shape)  // Softmax preserves shape
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testSoftmax_NegativeDimension_Success() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val result = ops.softmax(tensor, -1)  // Last dimension
        
        assertEquals(tensor.shape, result.shape)
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testSoftmax_InvalidDimension_ThrowsException() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.softmax(tensor, 5)  // Out of bounds
        }
        assertTrue(exception.message?.contains("out of bounds") == true)
    }
    
    @Test
    fun testSum_AllDimensions_Scalar() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val result = ops.sum(tensor, null)  // Sum all dimensions
        
        assertEquals(Shape(1), result.shape)  // Scalar result
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testSum_SpecificDimension_Success() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val result = ops.sum(tensor, 1)  // Sum along dimension 1
        
        assertEquals(Shape(2, 4), result.shape)  // Remove dimension 1
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testSum_LastDimension_Success() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val result = ops.sum(tensor, -1)  // Sum along last dimension
        
        assertEquals(Shape(2, 3), result.shape)  // Remove last dimension
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testSum_SingleDimensionTensor_Scalar() {
        val tensor = createTensor(Shape(5))
        
        val result = ops.sum(tensor, 0)  // Sum the only dimension
        
        assertEquals(Shape(1), result.shape)  // Result is scalar
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testMean_SpecificDimension_Success() {
        val tensor = createTensor(Shape(3, 4, 5))
        
        val result = ops.mean(tensor, 2)  // Mean along dimension 2
        
        assertEquals(Shape(3, 4), result.shape)  // Remove dimension 2
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testReduction_InvalidDimension_ThrowsException() {
        val tensor = createTensor(Shape(2, 3))
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.sum(tensor, 5)  // Out of bounds
        }
        assertTrue(exception.message?.contains("out of bounds") == true)
    }
    
    @Test
    fun testVariance_AllDimensions_Scalar() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val result = ops.variance(tensor, null)  // Variance over all dimensions
        
        assertEquals(Shape(1), result.shape)  // Scalar result
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testVariance_SpecificDimension_Success() {
        val tensor = createTensor(Shape(3, 4, 5))
        
        val result = ops.variance(tensor, 1)  // Variance along dimension 1
        
        assertEquals(Shape(3, 5), result.shape)  // Remove dimension 1
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testVariance_LastDimension_Success() {
        val tensor = createTensor(Shape(2, 3, 4))
        
        val result = ops.variance(tensor, -1)  // Variance along last dimension
        
        assertEquals(Shape(2, 3), result.shape)  // Remove last dimension
        assertEquals(FP32::class, result.dtype)
    }
    
    @Test
    fun testMathematicalFunctions_PreserveShape() {
        val tensor = createTensor(Shape(3, 4, 2))
        
        val sqrtResult = ops.sqrt(tensor)
        
        assertEquals(tensor.shape, sqrtResult.shape)  // sqrt preserves shape
        assertEquals(FP32::class, sqrtResult.dtype)
    }
    
    @Test
    fun testConvert_PreservesShape() {
        val floatTensor = createTensor(Shape(2, 3))
        
        val result = ops.convert(floatTensor, Int32)
        
        assertEquals(Shape(2, 3), result.shape)
        assertEquals(Int32::class, result.dtype)
    }
    
    @Test
    fun testEdgeCases_EmptyTensor() {
        val emptyTensor = createTensor(Shape(0))
        
        // Operations should work with empty tensors
        val reluResult = ops.relu(emptyTensor)
        assertEquals(Shape(0), reluResult.shape)
        
        val sumResult = ops.sum(emptyTensor, null)
        assertEquals(Shape(1), sumResult.shape)
    }
    
    @Test
    fun testEdgeCases_SingleElementTensor() {
        val scalarTensor = createTensor(Shape(1))
        
        // Test element-wise operations with scalars
        val addResult = ops.add(scalarTensor, scalarTensor)
        assertEquals(Shape(1), addResult.shape)
        
        val sumResult = ops.sum(scalarTensor, null)
        assertEquals(Shape(1), sumResult.shape)
    }
    
    @Test
    fun testMatmul_BatchDimensionBroadcasting() {
        val a = createTensor(Shape(1, 2, 3))  // Batch=1, 2x3
        val b = createTensor(Shape(4, 3, 5))  // Batch=4, 3x5
        
        val result = ops.matmul(a, b)
        
        assertEquals(Shape(4, 2, 5), result.shape)  // Broadcast batch dimension
    }
    
    @Test
    fun testMatmul_IncompatibleBatchDimensions_ThrowsException() {
        val a = createTensor(Shape(2, 3, 4))  // Batch=2
        val b = createTensor(Shape(3, 4, 5))  // Batch=3 (incompatible)
        
        val exception = assertFailsWith<IllegalArgumentException> {
            ops.matmul(a, b)
        }
        assertTrue(exception.message?.contains("batch dimension mismatch") == true)
    }
    
    @Test
    fun testLargeTensorShapes() {
        // Test with larger tensor shapes
        val largeTensor = createTensor(Shape(100, 200))
        
        val transposeResult = ops.transpose(largeTensor)
        assertEquals(Shape(200, 100), transposeResult.shape)
        
        val reshapeResult = ops.reshape(largeTensor, Shape(20, 1000))
        assertEquals(Shape(20, 1000), reshapeResult.shape)
    }
}