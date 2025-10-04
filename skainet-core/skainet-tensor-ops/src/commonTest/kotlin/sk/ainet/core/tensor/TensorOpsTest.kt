package sk.ainet.core.tensor

import kotlin.test.*

/**
 * Test suite for TensorOps interface using mock implementations.
 * This validates the interface contract and ensures all operations are properly defined.
 */
class TensorOpsTest {

    // Mock tensor class for testing
    private data class MockTensor(val value: Double, val mockShape: Shape) {
        override fun toString(): String = "MockTensor(value=$value, shape=$mockShape)"
    }

    // Mock TensorOps implementation for testing
    private class MockTensorOps : TensorOps<FP32, Double, MockTensor> {
        override fun matmul(a: MockTensor, b: MockTensor): MockTensor {
            return MockTensor(a.value * b.value, Shape(2, 2))
        }

        override fun matmul4d(a: MockTensor, b: MockTensor): MockTensor {
            return MockTensor(a.value * b.value, Shape(1, 2, 2, 2))
        }

        override fun scale(a: MockTensor, scalar: Double): MockTensor {
            return MockTensor(a.value * scalar, a.mockShape)
        }

        override fun dot(a: MockTensor, b: MockTensor): Double {
            return a.value * b.value
        }

        // Tensor-Tensor operations
        override fun MockTensor.plus(other: MockTensor): MockTensor {
            return MockTensor(this.value + other.value, this.mockShape)
        }

        override fun MockTensor.minus(other: MockTensor): MockTensor {
            return MockTensor(this.value - other.value, this.mockShape)
        }

        override fun MockTensor.times(other: MockTensor): MockTensor {
            return MockTensor(this.value * other.value, this.mockShape)
        }

        override fun MockTensor.div(other: MockTensor): MockTensor {
            return MockTensor(this.value / other.value, this.mockShape)
        }

        // Tensor-Scalar operations
        override fun MockTensor.plus(scalar: Int): MockTensor {
            return MockTensor(this.value + scalar, this.mockShape)
        }

        override fun MockTensor.minus(scalar: Int): MockTensor {
            return MockTensor(this.value - scalar, this.mockShape)
        }

        override fun MockTensor.times(scalar: Int): MockTensor {
            return MockTensor(this.value * scalar, this.mockShape)
        }

        override fun MockTensor.div(scalar: Int): MockTensor {
            return MockTensor(this.value / scalar, this.mockShape)
        }

        override fun MockTensor.plus(scalar: Float): MockTensor {
            return MockTensor(this.value + scalar, this.mockShape)
        }

        override fun MockTensor.minus(scalar: Float): MockTensor {
            return MockTensor(this.value - scalar, this.mockShape)
        }

        override fun MockTensor.times(scalar: Float): MockTensor {
            return MockTensor(this.value * scalar, this.mockShape)
        }

        override fun MockTensor.div(scalar: Float): MockTensor {
            return MockTensor(this.value / scalar, this.mockShape)
        }

        override fun MockTensor.plus(scalar: Double): MockTensor {
            return MockTensor(this.value + scalar, this.mockShape)
        }

        override fun MockTensor.minus(scalar: Double): MockTensor {
            return MockTensor(this.value - scalar, this.mockShape)
        }

        override fun MockTensor.times(scalar: Double): MockTensor {
            return MockTensor(this.value * scalar, this.mockShape)
        }

        override fun MockTensor.div(scalar: Double): MockTensor {
            return MockTensor(this.value / scalar, this.mockShape)
        }

        // Scalar-Tensor operations (using default implementations)
        override fun Double.plus(t: MockTensor): MockTensor {
            return MockTensor(this + t.value, t.mockShape)
        }

        override fun Double.minus(t: MockTensor): MockTensor {
            return MockTensor(this - t.value, t.mockShape)
        }

        override fun Double.times(t: MockTensor): MockTensor {
            return MockTensor(this * t.value, t.mockShape)
        }

        override fun Double.div(t: MockTensor): MockTensor {
            return MockTensor(this / t.value, t.mockShape)
        }

        // Mathematical functions
        override fun MockTensor.t(): MockTensor {
            // Simple transpose mock - reverse dimensions
            val dims = mockShape.dimensions
            val transposedDims = dims.reversedArray()
            return MockTensor(this.value, Shape(transposedDims))
        }

        override fun MockTensor.relu(): MockTensor {
            return MockTensor(maxOf(0.0, this.value), this.mockShape)
        }

        override fun MockTensor.softmax(dimension: Int): MockTensor {
            // Simple softmax mock - normalize to range [0,1]
            val exp = kotlin.math.exp(this.value)
            return MockTensor(exp / (exp + 1.0), this.mockShape)
        }

        override fun MockTensor.sigmoid(): MockTensor {
            val exp = kotlin.math.exp(-this.value)
            return MockTensor(1.0 / (1.0 + exp), this.mockShape)
        }

        override fun MockTensor.tanh(): MockTensor {
            return MockTensor(kotlin.math.tanh(this.value), this.mockShape)
        }

        override fun MockTensor.flatten(startDim: Int, endDim: Int): MockTensor {
            // Simple flatten mock - return 1D shape with same volume
            return MockTensor(this.value, Shape(this.mockShape.volume))
        }

        override fun MockTensor.reshape(newShape: Shape): MockTensor {
            require(newShape.volume == this.mockShape.volume) {
                "Cannot reshape tensor with volume ${this.mockShape.volume} to shape with volume ${newShape.volume}"
            }
            return MockTensor(this.value, newShape)
        }

        override fun MockTensor.reshape(vararg dimensions: Int): MockTensor {
            val newShape = Shape(*dimensions)
            return reshape(newShape)
        }
    }

    private val mockOps = MockTensorOps()

    @Test
    fun testTensorOpsMatmul() {
        val tensor1 = MockTensor(2.0, Shape(2, 3))
        val tensor2 = MockTensor(3.0, Shape(3, 2))
        
        val result = mockOps.matmul(tensor1, tensor2)
        
        assertEquals(6.0, result.value)
        assertEquals(Shape(2, 2), result.mockShape)
    }

    @Test
    fun testTensorOpsMatmul4d() {
        val tensor1 = MockTensor(2.0, Shape(1, 2, 3, 4))
        val tensor2 = MockTensor(3.0, Shape(1, 2, 4, 5))
        
        val result = mockOps.matmul4d(tensor1, tensor2)
        
        assertEquals(6.0, result.value)
        assertEquals(Shape(1, 2, 2, 2), result.mockShape)
    }

    @Test
    fun testTensorOpsScale() {
        val tensor = MockTensor(5.0, Shape(2, 3))
        
        val result = mockOps.scale(tensor, 2.0)
        
        assertEquals(10.0, result.value)
        assertEquals(Shape(2, 3), result.mockShape)
    }

    @Test
    fun testTensorOpsDot() {
        val tensor1 = MockTensor(4.0, Shape(3))
        val tensor2 = MockTensor(2.0, Shape(3))
        
        val result = mockOps.dot(tensor1, tensor2)
        
        assertEquals(8.0, result)
    }

    @Test
    fun testTensorTensorOperations() {
        val tensor1 = MockTensor(10.0, Shape(2, 2))
        val tensor2 = MockTensor(3.0, Shape(2, 2))
        
        with(mockOps) {
            // Test tensor-tensor arithmetic
            val plusResult = tensor1.plus(tensor2)
            assertEquals(13.0, plusResult.value)
            assertEquals(Shape(2, 2), plusResult.mockShape)
            
            val minusResult = tensor1.minus(tensor2)
            assertEquals(7.0, minusResult.value)
            
            val timesResult = tensor1.times(tensor2)
            assertEquals(30.0, timesResult.value)
            
            val divResult = tensor1.div(tensor2)
            assertEquals(10.0 / 3.0, divResult.value, 0.001)
        }
    }

    @Test
    fun testTensorScalarOperations() {
        val tensor = MockTensor(10.0, Shape(2, 2))
        
        with(mockOps) {
            // Test tensor-int operations
            assertEquals(15.0, tensor.plus(5).value)
            assertEquals(5.0, tensor.minus(5).value)
            assertEquals(50.0, tensor.times(5).value)
            assertEquals(2.0, tensor.div(5).value)
            
            // Test tensor-float operations
            assertEquals(12.5, tensor.plus(2.5f).value, 0.001)
            assertEquals(7.5, tensor.minus(2.5f).value, 0.001)
            assertEquals(25.0, tensor.times(2.5f).value, 0.001)
            assertEquals(4.0, tensor.div(2.5f).value, 0.001)
            
            // Test tensor-double operations
            assertEquals(13.5, tensor.plus(3.5).value, 0.001)
            assertEquals(6.5, tensor.minus(3.5).value, 0.001)
            assertEquals(35.0, tensor.times(3.5).value, 0.001)
            assertEquals(2.857, tensor.div(3.5).value, 0.001)
        }
    }

    @Test
    fun testScalarTensorOperations() {
        val tensor = MockTensor(4.0, Shape(2, 2))
        
        with(mockOps) {
            // Test double-tensor operations
            assertEquals(14.0, (10.0).plus(tensor).value)
            assertEquals(6.0, (10.0).minus(tensor).value)
            assertEquals(40.0, (10.0).times(tensor).value)
            assertEquals(2.5, (10.0).div(tensor).value)
            
            // Test default int-tensor operations (should delegate to double)
            assertEquals(9.0, (5).plus(tensor).value)
            assertEquals(1.0, (5).minus(tensor).value)
            assertEquals(20.0, (5).times(tensor).value)
            assertEquals(1.25, (5).div(tensor).value)
            
            // Test default float-tensor operations (should delegate to double)
            assertEquals(6.5, (2.5f).plus(tensor).value, 0.001)
            assertEquals(-1.5, (2.5f).minus(tensor).value, 0.001)
            assertEquals(10.0, (2.5f).times(tensor).value, 0.001)
            assertEquals(0.625, (2.5f).div(tensor).value, 0.001)
        }
    }

    @Test
    fun testMathematicalFunctions() {
        val tensor = MockTensor(2.0, Shape(2, 3))
        
        with(mockOps) {
            // Test transpose
            val transposed = tensor.t()
            assertEquals(2.0, transposed.value)
            assertEquals(Shape(3, 2), transposed.mockShape) // Dimensions reversed
            
            // Test ReLU
            val positiveTensor = MockTensor(5.0, Shape(2, 2))
            val negativeTensor = MockTensor(-3.0, Shape(2, 2))
            assertEquals(5.0, positiveTensor.relu().value)
            assertEquals(0.0, negativeTensor.relu().value)
            
            // Test sigmoid
            val sigmoidResult = tensor.sigmoid()
            assertTrue(sigmoidResult.value > 0.0 && sigmoidResult.value < 1.0)
            assertEquals(Shape(2, 3), sigmoidResult.mockShape)
            
            // Test tanh
            val tanhResult = tensor.tanh()
            assertTrue(tanhResult.value > -1.0 && tanhResult.value < 1.0)
            assertEquals(Shape(2, 3), tanhResult.mockShape)
            
            // Test softmax
            val softmaxResult = tensor.softmax(0)
            assertTrue(softmaxResult.value > 0.0 && softmaxResult.value < 1.0)
            assertEquals(Shape(2, 3), softmaxResult.mockShape)
        }
    }

    @Test
    fun testReshapeOperations() {
        val tensor = MockTensor(5.0, Shape(2, 3)) // Volume = 6
        
        with(mockOps) {
            // Test reshape with Shape
            val reshaped1 = tensor.reshape(Shape(3, 2))
            assertEquals(5.0, reshaped1.value)
            assertEquals(Shape(3, 2), reshaped1.mockShape)
            
            val reshaped2 = tensor.reshape(Shape(6))
            assertEquals(5.0, reshaped2.value)
            assertEquals(Shape(6), reshaped2.mockShape)
            
            // Test reshape with varargs
            val reshaped3 = tensor.reshape(1, 6)
            assertEquals(5.0, reshaped3.value)
            assertEquals(Shape(1, 6), reshaped3.mockShape)
            
            val reshaped4 = tensor.reshape(2, 3, 1)
            assertEquals(5.0, reshaped4.value)
            assertEquals(Shape(2, 3, 1), reshaped4.mockShape)
            
            // Test invalid reshape
            assertFailsWith<IllegalArgumentException> {
                tensor.reshape(Shape(2, 4)) // Volume = 8, doesn't match original volume = 6
            }
        }
    }

    @Test
    fun testFlattenOperation() {
        val tensor = MockTensor(7.0, Shape(2, 3, 4)) // Volume = 24
        
        with(mockOps) {
            val flattened = tensor.flatten()
            assertEquals(7.0, flattened.value)
            assertEquals(Shape(24), flattened.mockShape)
        }
    }

    @Test
    fun testTensorOpsInterfaceCompleteness() {
        // This test ensures all TensorOps methods are callable through the interface
        val tensor1 = MockTensor(2.0, Shape(2, 2))
        val tensor2 = MockTensor(3.0, Shape(2, 2))
        val ops: TensorOps<FP32, Double, MockTensor> = mockOps
        
        // Test that all operations are accessible through the interface
        assertNotNull(ops.matmul(tensor1, tensor2))
        assertNotNull(ops.matmul4d(tensor1, tensor2))
        assertNotNull(ops.scale(tensor1, 2.0))
        assertNotNull(ops.dot(tensor1, tensor2))
        
        with(ops) {
            // Extension functions should be accessible
            assertNotNull(tensor1.plus(tensor2))
            assertNotNull(tensor1.minus(tensor2))
            assertNotNull(tensor1.times(tensor2))
            assertNotNull(tensor1.div(tensor2))
            
            // Scalar operations
            assertNotNull(tensor1.plus(1))
            assertNotNull(tensor1.plus(1.0f))
            assertNotNull(tensor1.plus(1.0))
            
            // Mathematical functions
            assertNotNull(tensor1.t())
            assertNotNull(tensor1.relu())
            assertNotNull(tensor1.sigmoid())
            assertNotNull(tensor1.tanh())
            assertNotNull(tensor1.softmax(0))
            assertNotNull(tensor1.flatten())
            assertNotNull(tensor1.reshape(Shape(4)))
            assertNotNull(tensor1.reshape(2, 2))
        }
    }
}