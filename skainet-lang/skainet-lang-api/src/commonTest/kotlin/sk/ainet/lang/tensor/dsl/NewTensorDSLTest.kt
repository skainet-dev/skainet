package sk.ainet.lang.tensor.dsl

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.FP32
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Test for the new modern DSL syntax
 */
class NewTensorDSLTest {
    
    // Mock TensorData for testing
    private class MockTensorData(
        override val shape: Shape,
        private val data: FloatArray
    ) : TensorData<FP32, Float> {
        override fun get(vararg indices: Int): Float = data[shape.index(indices)]
        override fun set(vararg indices: Int, value: Float) { data[shape.index(indices)] = value }
    }

    // Simple mock factory for testing
    private val testFactory = object : TensorDataFactory<FP32, Float> {
        override fun zeros(shape: Shape, dtype: FP32): TensorData<FP32, Float> = 
            MockTensorData(shape, FloatArray(shape.volume) { 0.0f })
            
        override fun ones(shape: Shape, dtype: FP32): TensorData<FP32, Float> = 
            MockTensorData(shape, FloatArray(shape.volume) { 1.0f })
            
        override fun full(shape: Shape, value: Number, dtype: FP32): TensorData<FP32, Float> = 
            MockTensorData(shape, FloatArray(shape.volume) { value.toFloat() })
            
        override fun randn(shape: Shape, mean: Float, std: Float, dtype: FP32, random: Random): TensorData<FP32, Float> = 
            MockTensorData(shape, FloatArray(shape.volume) { 
                // Simple normal distribution approximation
                val u1 = random.nextFloat()
                val u2 = random.nextFloat()
                val z = kotlin.math.sqrt(-2.0 * kotlin.math.ln(u1.toDouble())) * kotlin.math.cos(2.0 * kotlin.math.PI * u2.toDouble())
                (z.toFloat() * std + mean)
            })
            
        override fun uniform(shape: Shape, min: Float, max: Float, dtype: FP32, random: Random): TensorData<FP32, Float> = 
            MockTensorData(shape, FloatArray(shape.volume) { random.nextFloat() * (max - min) + min })
            
        override fun init(shape: Shape, dtype: FP32, generator: (indices: IntArray) -> Float): TensorData<FP32, Float> = 
            MockTensorData(shape, FloatArray(shape.volume) { index ->
                val indices = computeIndices(index, shape)
                generator(indices)
            })
            
        override fun randomInit(shape: Shape, dtype: FP32, generator: (random: Random) -> Float, random: Random): TensorData<FP32, Float> = 
            MockTensorData(shape, FloatArray(shape.volume) { generator(random) })
            
        private fun computeIndices(flatIndex: Int, shape: Shape): IntArray {
            val indices = IntArray(shape.rank)
            var remaining = flatIndex
            for (i in shape.rank - 1 downTo 0) {
                indices[i] = remaining % shape[i]
                remaining /= shape[i]
            }
            return indices
        }
    }

    @Test
    fun testNewDSLSyntaxWithRandomInit() {
        // Test the new DSL syntax as specified in the issue
        with(testFactory) {
            val matrix = tensor(2, 2) { shape ->
                randomInit({ random: Random -> 
                    if (random.nextBoolean()) 1.0f else -1.0f // Random sign
                })
            }
            
            assertNotNull(matrix)
            assertEquals(Shape(2, 2), matrix.shape)
            assertEquals(4, matrix.volume)
            
            // Check that values are either 1.0f or -1.0f
            for (i in 0 until 2) {
                for (j in 0 until 2) {
                    val value = matrix.data[i, j]
                    assertTrue(value == 1.0f || value == -1.0f, "Value should be 1.0f or -1.0f, but was $value")
                }
            }
        }
    }

    @Test
    fun testNewDSLSyntaxWithZeros() {
        with(testFactory) {
            val matrix = tensor(3, 3) { shape ->
                zeros()
            }
            
            assertNotNull(matrix)
            assertEquals(Shape(3, 3), matrix.shape)
            assertEquals(0.0f, matrix.data[0, 0])
            assertEquals(0.0f, matrix.data[1, 1])
            assertEquals(0.0f, matrix.data[2, 2])
        }
    }

    @Test
    fun testNewDSLSyntaxWithOnes() {
        with(testFactory) {
            val vector = tensor(5) { shape ->
                ones()
            }
            
            assertNotNull(vector)
            assertEquals(Shape(5), vector.shape)
            assertEquals(1.0f, vector.data[0])
            assertEquals(1.0f, vector.data[4])
        }
    }
}