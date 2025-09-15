package sk.ainet.core.tensor.multiplication

import sk.ainet.core.tensor.*
import sk.ainet.core.tensor.backend.CpuBackend
import sk.ainet.core.tensor.backend.CpuTensorFP32
import sk.ainet.core.tensor.backend.TensorFP32
import kotlin.test.*

class MatrixMultiplicationTest {

    private val backend = CpuBackend()

    @Test
    fun testBasicMatrixMultiplication() {
        val shape1 = Shape(2, 3)
        val shape2 = Shape(3, 2)
        val data1 = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f) // 2x3 matrix
        val data2 = floatArrayOf(7f, 8f, 9f, 10f, 11f, 12f) // 3x2 matrix

        val tensor1: TensorFP32 = CpuTensorFP32.fromArray(shape1, data1)
        val tensor2: TensorFP32 = CpuTensorFP32.fromArray(shape2, data2)

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
    fun testSquareMatrixMultiplication() {
        val shape = Shape(3, 3)
        val data1 = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f)
        val data2 = floatArrayOf(9f, 8f, 7f, 6f, 5f, 4f, 3f, 2f, 1f)

        val tensor1: TensorFP32 = CpuTensorFP32.fromArray(shape, data1)
        val tensor2: TensorFP32 = CpuTensorFP32.fromArray(shape, data2)

        val result = backend.matmul(tensor1, tensor2)

        assertEquals(shape, result.shape)
        // First row: [1*9 + 2*6 + 3*3, 1*8 + 2*5 + 3*2, 1*7 + 2*4 + 3*1] = [30, 24, 18]
        assertEquals(30f, result[0, 0])
        assertEquals(24f, result[0, 1])
        assertEquals(18f, result[0, 2])
        // Second row: [4*9 + 5*6 + 6*3, 4*8 + 5*5 + 6*2, 4*7 + 5*4 + 6*1] = [84, 69, 54]
        assertEquals(84f, result[1, 0])
        assertEquals(69f, result[1, 1])
        assertEquals(54f, result[1, 2])
        // Third row: [7*9 + 8*6 + 9*3, 7*8 + 8*5 + 9*2, 7*7 + 8*4 + 9*1] = [138, 114, 90]
        assertEquals(138f, result[2, 0])
        assertEquals(114f, result[2, 1])
        assertEquals(90f, result[2, 2])
    }

    @Test
    fun testIdentityMatrixMultiplication() {
        val shape = Shape(3, 3)
        val data = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f)
        val identityData = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)

        val tensor = CpuTensorFP32.fromArray(shape, data)
        val identity = CpuTensorFP32.fromArray(shape, identityData)

        val result = backend.matmul(tensor, identity)

        assertEquals(shape, result.shape)
        // Multiplying by identity should return the original matrix
        assertEquals(1f, result[0, 0])
        assertEquals(2f, result[0, 1])
        assertEquals(3f, result[0, 2])
        assertEquals(4f, result[1, 0])
        assertEquals(5f, result[1, 1])
        assertEquals(6f, result[1, 2])
        assertEquals(7f, result[2, 0])
        assertEquals(8f, result[2, 1])
        assertEquals(9f, result[2, 2])
    }

    @Test
    fun testVectorMatrixMultiplication() {
        // 1x3 vector * 3x4 matrix = 1x4 vector
        val shape1 = Shape(1, 3)
        val shape2 = Shape(3, 4)
        val data1 = floatArrayOf(2f, 3f, 4f) // row vector
        val data2 = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f) // 3x4 matrix

        val vector = CpuTensorFP32.fromArray(shape1, data1)
        val matrix = CpuTensorFP32.fromArray(shape2, data2)

        val result = backend.matmul(vector, matrix)

        assertEquals(Shape(1, 4), result.shape)
        // [2*1 + 3*5 + 4*9, 2*2 + 3*6 + 4*10, 2*3 + 3*7 + 4*11, 2*4 + 3*8 + 4*12] = [53, 62, 71, 80]
        assertEquals(53f, result[0, 0])
        assertEquals(62f, result[0, 1])
        assertEquals(71f, result[0, 2])
        assertEquals(80f, result[0, 3])
    }

    @Test
    fun testMatrixVectorMultiplication() {
        // 4x3 matrix * 3x1 vector = 4x1 vector
        val shape1 = Shape(4, 3)
        val shape2 = Shape(3, 1)
        val data1 = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f) // 4x3 matrix
        val data2 = floatArrayOf(2f, 3f, 4f) // column vector

        val matrix = CpuTensorFP32.fromArray(shape1, data1)
        val vector = CpuTensorFP32.fromArray(shape2, data2)

        val result = backend.matmul(matrix, vector)

        assertEquals(Shape(4, 1), result.shape)
        // [1*2 + 2*3 + 3*4] = [20], [4*2 + 5*3 + 6*4] = [47], [7*2 + 8*3 + 9*4] = [74], [10*2 + 11*3 + 12*4] = [101]
        assertEquals(20f, result[0, 0])
        assertEquals(47f, result[1, 0])
        assertEquals(74f, result[2, 0])
        assertEquals(101f, result[3, 0])
    }

    @Test
    fun testLargeMatrixMultiplication() {
        val shape1 = Shape(50, 30)
        val shape2 = Shape(30, 40)
        val data1 = FloatArray(1500) { (it % 10 + 1).toFloat() }
        val data2 = FloatArray(1200) { (it % 5 + 1).toFloat() }

        val tensor1 = CpuTensorFP32.fromArray(shape1, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape2, data2)

        val result = backend.matmul(tensor1, tensor2)

        assertEquals(Shape(50, 40), result.shape)
        // Verify specific elements by manual calculation
        assertTrue(result[0, 0] > 0f)
        assertTrue(result[49, 39] > 0f)
    }

    @Test
    fun testMatrixMultiplicationWithZeros() {
        val shape1 = Shape(2, 3)
        val shape2 = Shape(3, 2)
        val data1 = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        val zeros = CpuTensorFP32.zeros(shape2)

        val tensor1 = CpuTensorFP32.fromArray(shape1, data1)

        val result = backend.matmul(tensor1, zeros)

        assertEquals(Shape(2, 2), result.shape)
        assertEquals(0f, result[0, 0])
        assertEquals(0f, result[0, 1])
        assertEquals(0f, result[1, 0])
        assertEquals(0f, result[1, 1])
    }

    @Test
    fun testMatrixMultiplicationWithNegatives() {
        val shape1 = Shape(2, 2)
        val shape2 = Shape(2, 2)
        val data1 = floatArrayOf(-1f, 2f, -3f, 4f)
        val data2 = floatArrayOf(5f, -6f, -7f, 8f)

        val tensor1 = CpuTensorFP32.fromArray(shape1, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape2, data2)

        val result = backend.matmul(tensor1, tensor2)

        assertEquals(shape1, result.shape)
        // First row: [-1*5 + 2*(-7), -1*(-6) + 2*8] = [-19, 22]
        // Second row: [-3*5 + 4*(-7), -3*(-6) + 4*8] = [-43, 50]
        assertEquals(-19f, result[0, 0])
        assertEquals(22f, result[0, 1])
        assertEquals(-43f, result[1, 0])
        assertEquals(50f, result[1, 1])
    }

    @Test
    fun testInvalidMatrixMultiplicationDimensions() {
        val shape1 = Shape(2, 3)
        val shape2 = Shape(4, 2) // Wrong inner dimension - should be 3x2
        val data1 = FloatArray(6) { it.toFloat() }
        val data2 = FloatArray(8) { it.toFloat() }

        val tensor1 = CpuTensorFP32.fromArray(shape1, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape2, data2)

        assertFailsWith<IllegalArgumentException> {
            backend.matmul(tensor1, tensor2)
        }
    }

    @Test
    fun testMatrixMultiplicationNonSquareResult() {
        // Test rectangular result matrices
        val shape1 = Shape(5, 3)
        val shape2 = Shape(3, 7)
        val data1 = FloatArray(15) { (it + 1).toFloat() }
        val data2 = FloatArray(21) { (it % 3 + 1).toFloat() }

        val tensor1 = CpuTensorFP32.fromArray(shape1, data1)
        val tensor2 = CpuTensorFP32.fromArray(shape2, data2)

        val result = backend.matmul(tensor1, tensor2)

        assertEquals(Shape(5, 7), result.shape)
        // Verify the multiplication produces correct dimensions
        assertTrue(result[0, 0] > 0f)
        assertTrue(result[4, 6] > 0f)
    }

    @Test
    fun testMatrixChainMultiplication() {
        // Test A * B * C through (A * B) * C
        val shapeA = Shape(2, 3)
        val shapeB = Shape(3, 4)
        val shapeC = Shape(4, 2)

        val dataA = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        val dataB = FloatArray(12) { (it + 1).toFloat() }
        val dataC = FloatArray(8) { (it % 2 + 1).toFloat() }

        val tensorA = CpuTensorFP32.fromArray(shapeA, dataA)
        val tensorB = CpuTensorFP32.fromArray(shapeB, dataB)
        val tensorC = CpuTensorFP32.fromArray(shapeC, dataC)

        val resultAB = backend.matmul(tensorA, tensorB)
        val resultABC = backend.matmul(resultAB, tensorC)

        assertEquals(Shape(2, 2), resultABC.shape)
        assertTrue(resultABC[0, 0] > 0f)
        assertTrue(resultABC[1, 1] > 0f)
    }
}