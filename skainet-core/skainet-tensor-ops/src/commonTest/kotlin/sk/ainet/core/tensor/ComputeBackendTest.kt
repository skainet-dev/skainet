package sk.ainet.core.tensor

import sk.ainet.core.tensor.backend.ComputeBackend
import kotlin.test.*

/**
 * Test suite for ComputeBackend interface using mock implementations.
 * This validates the backend interface contract and ensures proper separation of concerns.
 */
class ComputeBackendTest {

    // Mock tensor class for testing
    private data class MockTensor(val value: Double, val mockShape: Shape) {
        override fun toString(): String = "MockTensor(value=$value, shape=$mockShape)"
    }

    // Mock ComputeBackend implementation for testing
    private class MockComputeBackend : ComputeBackend<FP32, Double, MockTensor> {
        
        override val name: String = "MockBackend"

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

        // Scalar-Tensor operations
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
            val dims = mockShape.dimensions
            val transposedDims = dims.reversedArray()
            return MockTensor(this.value, Shape(transposedDims))
        }

        override fun MockTensor.relu(): MockTensor {
            return MockTensor(maxOf(0.0, this.value), this.mockShape)
        }

        override fun MockTensor.softmax(dimension: Int): MockTensor {
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

    private val mockBackend = MockComputeBackend()

    @Test
    fun testComputeBackendName() {
        assertEquals("MockBackend", mockBackend.name)
    }

    @Test
    fun testComputeBackendMatmul() {
        val tensor1 = MockTensor(3.0, Shape(2, 3))
        val tensor2 = MockTensor(4.0, Shape(3, 2))
        
        val result = mockBackend.matmul(tensor1, tensor2)
        
        assertEquals(12.0, result.value)
        assertEquals(Shape(2, 2), result.mockShape)
    }

    @Test
    fun testComputeBackendMatmul4d() {
        val tensor1 = MockTensor(2.5, Shape(1, 2, 3, 4))
        val tensor2 = MockTensor(4.0, Shape(1, 2, 4, 5))
        
        val result = mockBackend.matmul4d(tensor1, tensor2)
        
        assertEquals(10.0, result.value)
        assertEquals(Shape(1, 2, 2, 2), result.mockShape)
    }

    @Test
    fun testComputeBackendScale() {
        val tensor = MockTensor(3.0, Shape(2, 2))
        
        val result = mockBackend.scale(tensor, 4.0)
        
        assertEquals(12.0, result.value)
        assertEquals(Shape(2, 2), result.mockShape)
    }

    @Test
    fun testComputeBackendDot() {
        val tensor1 = MockTensor(5.0, Shape(3))
        val tensor2 = MockTensor(2.0, Shape(3))
        
        val result = mockBackend.dot(tensor1, tensor2)
        
        assertEquals(10.0, result)
    }

    @Test
    fun testComputeBackendTensorOperations() {
        val tensor1 = MockTensor(8.0, Shape(2, 2))
        val tensor2 = MockTensor(2.0, Shape(2, 2))
        
        with(mockBackend) {
            // Test tensor-tensor arithmetic
            assertEquals(10.0, tensor1.plus(tensor2).value)
            assertEquals(6.0, tensor1.minus(tensor2).value)
            assertEquals(16.0, tensor1.times(tensor2).value)
            assertEquals(4.0, tensor1.div(tensor2).value)
            
            // Test tensor-scalar operations
            assertEquals(13.0, tensor1.plus(5).value)
            assertEquals(3.0, tensor1.minus(5).value)
            assertEquals(40.0, tensor1.times(5).value)
            assertEquals(1.6, tensor1.div(5).value)
            
            assertEquals(10.5, tensor1.plus(2.5f).value, 0.001)
            assertEquals(5.5, tensor1.minus(2.5f).value, 0.001)
            assertEquals(20.0, tensor1.times(2.5f).value, 0.001)
            assertEquals(3.2, tensor1.div(2.5f).value, 0.001)
            
            assertEquals(11.5, tensor1.plus(3.5).value, 0.001)
            assertEquals(4.5, tensor1.minus(3.5).value, 0.001)
            assertEquals(28.0, tensor1.times(3.5).value, 0.001)
            assertEquals(2.286, tensor1.div(3.5).value, 0.001)
        }
    }

    @Test
    fun testComputeBackendScalarTensorOperations() {
        val tensor = MockTensor(4.0, Shape(2, 2))
        
        with(mockBackend) {
            // Test scalar-tensor operations
            assertEquals(16.0, (12.0).plus(tensor).value)
            assertEquals(8.0, (12.0).minus(tensor).value)
            assertEquals(48.0, (12.0).times(tensor).value)
            assertEquals(3.0, (12.0).div(tensor).value)
            
            // Test default int operations (delegate to double)
            assertEquals(9.0, (5).plus(tensor).value)
            assertEquals(1.0, (5).minus(tensor).value)
            assertEquals(20.0, (5).times(tensor).value)
            assertEquals(1.25, (5).div(tensor).value)
            
            // Test default float operations (delegate to double)
            assertEquals(6.5, (2.5f).plus(tensor).value, 0.001)
            assertEquals(-1.5, (2.5f).minus(tensor).value, 0.001)
            assertEquals(10.0, (2.5f).times(tensor).value, 0.001)
            assertEquals(0.625, (2.5f).div(tensor).value, 0.001)
        }
    }

    @Test
    fun testComputeBackendMathematicalFunctions() {
        val tensor = MockTensor(1.5, Shape(3, 2))
        
        with(mockBackend) {
            // Test transpose
            val transposed = tensor.t()
            assertEquals(1.5, transposed.value)
            assertEquals(Shape(2, 3), transposed.mockShape)
            
            // Test ReLU
            val positiveTensor = MockTensor(3.0, Shape(2, 2))
            val negativeTensor = MockTensor(-2.0, Shape(2, 2))
            assertEquals(3.0, positiveTensor.relu().value)
            assertEquals(0.0, negativeTensor.relu().value)
            
            // Test sigmoid
            val sigmoidResult = tensor.sigmoid()
            assertTrue(sigmoidResult.value > 0.0 && sigmoidResult.value < 1.0)
            assertEquals(Shape(3, 2), sigmoidResult.mockShape)
            
            // Test tanh
            val tanhResult = tensor.tanh()
            assertTrue(tanhResult.value > -1.0 && tanhResult.value < 1.0)
            assertEquals(Shape(3, 2), tanhResult.mockShape)
            
            // Test softmax
            val softmaxResult = tensor.softmax(0)
            assertTrue(softmaxResult.value > 0.0 && softmaxResult.value < 1.0)
            assertEquals(Shape(3, 2), softmaxResult.mockShape)
        }
    }

    @Test
    fun testComputeBackendReshapeOperations() {
        val tensor = MockTensor(7.0, Shape(2, 6)) // Volume = 12
        
        with(mockBackend) {
            // Test reshape with Shape
            val reshaped1 = tensor.reshape(Shape(3, 4))
            assertEquals(7.0, reshaped1.value)
            assertEquals(Shape(3, 4), reshaped1.mockShape)
            
            val reshaped2 = tensor.reshape(Shape(12))
            assertEquals(7.0, reshaped2.value)
            assertEquals(Shape(12), reshaped2.mockShape)
            
            // Test reshape with varargs
            val reshaped3 = tensor.reshape(1, 12)
            assertEquals(7.0, reshaped3.value)
            assertEquals(Shape(1, 12), reshaped3.mockShape)
            
            val reshaped4 = tensor.reshape(2, 2, 3)
            assertEquals(7.0, reshaped4.value)
            assertEquals(Shape(2, 2, 3), reshaped4.mockShape)
            
            // Test invalid reshape
            assertFailsWith<IllegalArgumentException> {
                tensor.reshape(Shape(2, 5)) // Volume = 10, doesn't match original volume = 12
            }
        }
    }

    @Test
    fun testComputeBackendFlatten() {
        val tensor = MockTensor(4.5, Shape(2, 3, 2)) // Volume = 12
        
        with(mockBackend) {
            val flattened = tensor.flatten()
            assertEquals(4.5, flattened.value)
            assertEquals(Shape(12), flattened.mockShape)
        }
    }

    @Test
    fun testComputeBackendInterfaceInheritance() {
        // Test that ComputeBackend properly extends TensorOps
        val backend: TensorOps<FP32, Double, MockTensor> = mockBackend
        val tensor1 = MockTensor(2.0, Shape(2, 2))
        val tensor2 = MockTensor(3.0, Shape(2, 2))
        
        // All TensorOps methods should be available
        assertNotNull(backend.matmul(tensor1, tensor2))
        assertNotNull(backend.matmul4d(tensor1, tensor2))
        assertNotNull(backend.scale(tensor1, 2.0))
        assertNotNull(backend.dot(tensor1, tensor2))
        
        with(backend) {
            // Extension functions should work through TensorOps interface
            assertNotNull(tensor1.plus(tensor2))
            assertNotNull(tensor1.minus(tensor2))
            assertNotNull(tensor1.times(tensor2))
            assertNotNull(tensor1.div(tensor2))
            assertNotNull(tensor1.t())
            assertNotNull(tensor1.relu())
            assertNotNull(tensor1.sigmoid())
            assertNotNull(tensor1.tanh())
            assertNotNull(tensor1.softmax(0))
            assertNotNull(tensor1.flatten())
            assertNotNull(tensor1.reshape(Shape(4)))
        }
    }

    @Test
    fun testComputeBackendNameProperty() {
        // Test the name property specific to ComputeBackend
        val backend: ComputeBackend<FP32, Double, MockTensor> = mockBackend
        assertEquals("MockBackend", backend.name)
    }

    @Test
    fun testComputeBackendPolymorphism() {
        // Test polymorphic usage
        val backends: List<TensorOps<FP32, Double, MockTensor>> = listOf(mockBackend)
        val computeBackends: List<ComputeBackend<FP32, Double, MockTensor>> = listOf(mockBackend)
        
        // Should be usable as TensorOps
        for (backend in backends) {
            val tensor = MockTensor(1.0, Shape(2, 2))
            with(backend) {
                assertNotNull(tensor.plus(1.0))
            }
        }
        
        // Should be usable as ComputeBackend with name property
        for (backend in computeBackends) {
            assertNotNull(backend.name)
            assertEquals("MockBackend", backend.name)
        }
    }
}