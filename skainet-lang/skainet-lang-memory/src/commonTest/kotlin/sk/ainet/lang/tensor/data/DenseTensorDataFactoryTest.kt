package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.Int32
import kotlin.random.Random
import kotlin.test.*

class DenseTensorDataFactoryTest {

    private val factory = DenseTensorDataFactory()

    @Test
    fun testZerosInt32() {
        val shape = Shape(2, 3)
        val tensorData = factory.full(shape, 0,Int32)
        
        assertEquals(shape, tensorData.shape)
        assertEquals(0, tensorData[0, 0])
        assertEquals(0, tensorData[0, 1])
        assertEquals(0, tensorData[1, 2])
    }

    @Test
    fun testZerosFP32() {
        val shape = Shape(2, 2)
        val tensorData = factory.full(shape, 0, FP32)
        
        assertEquals(shape, tensorData.shape)
        assertEquals(0.0f, tensorData[0, 0] as Float)
        assertEquals(0.0f, tensorData[1, 1] as Float)
    }

    @Test
    fun testZerosFP16() {
        val shape = Shape(3, 1)
        val tensorData = factory.full(shape, 0, FP16)
        
        assertEquals(shape, tensorData.shape)
        assertEquals(0.0f, tensorData[0, 0] as Float)
        assertEquals(0.0f, tensorData[2, 0] as Float)
    }

    @Test
    fun testOnesInt32() {
        val shape = Shape(2, 2)
        val tensorData = factory.full(shape,1, Int32)
        
        assertEquals(shape, tensorData.shape)
        assertEquals(1, tensorData[0, 0])
        assertEquals(1, tensorData[1, 1])
    }

    @Test
    fun testOnesFP32() {
        val shape = Shape(3, 2)
        val tensorData = factory.full(shape, 1, FP32)
        
        assertEquals(shape, tensorData.shape)
        assertEquals(1.0f, tensorData[0, 0] as Float)
        assertEquals(1.0f, tensorData[2, 1] as Float)
    }

    @Test
    fun testFullInt32() {
        val shape = Shape(2, 3)
        val value = 42
        val tensorData = factory.full(shape, value, Int32)
        
        assertEquals(shape, tensorData.shape)
        assertEquals(42, tensorData[0, 0])
        assertEquals(42, tensorData[1, 2])
    }

    @Test
    fun testFullFP32() {
        val shape = Shape(2, 2)
        val value = 3.14f
        val tensorData = factory.full(shape, value, FP32)
        
        assertEquals(shape, tensorData.shape)
        assertEquals(3.14f, tensorData[0, 0] as Float, 0.001f)
        assertEquals(3.14f, tensorData[1, 1] as Float, 0.001f)
    }

    @Test
    fun testRandomInt32() {
        /*
        val shape = Shape(3, 3)
        val seed = 12345L
        val random = Random(seed)
        val tensorData = factory .random(shape, Int32, random)
        
        assertEquals(shape, tensorData.shape)
        
        // Values should be in range [0, 100)
        val value = tensorData[0, 0] as Int
        assertTrue(value >= 0 && value < 100, "Random int value should be in range [0, 100)")

         */
    }

    @Test
    fun testRandomFP32() {
        val shape = Shape(2, 2)
        val seed = 54321L
        val random = Random(seed)
        val tensorData = factory.randn(shape,0.3f, 2.0f, FP32, random)
        
        assertEquals(shape, tensorData.shape)
        
        // Values should be in range [0, 1)
        val value = tensorData[0, 0] as Float
        assertTrue(value >= 0.0f && value < 1.0f, "Random float value should be in range [0, 1)")
    }

    @Test
    fun testRandnFP32() {
        val shape = Shape(4, 4)
        val mean = 2.0f
        val std = 0.5f
        val seed = 98765L
        val random = Random(seed)
        val tensorData = factory.randn(shape, mean, std, FP32, random)
        
        assertEquals(shape, tensorData.shape)
        
        // Check that values are generated (we can't easily test normality in unit tests)
        val value = tensorData[0, 0] as Float
        assertNotEquals(0.0f, value)
    }

    @Test
    fun testRandnDefaultParameters() {
        val shape = Shape(2, 2)
        val tensorData = factory.randn(shape, dtype = FP32)
        
        assertEquals(shape, tensorData.shape)
        
        // Should generate non-zero values with default mean=0, std=1
        val value = tensorData[0, 0] as Float
        // Just check it's a valid float (not NaN)
        assertFalse(value.isNaN(), "Generated value should not be NaN")
    }

    @Test
    fun testRandnNotSupportedForInt32() {
        val shape = Shape(2, 2)
        
        val exception = assertFailsWith<IllegalArgumentException> {
            factory.randn(shape, dtype = Int32)
        }
        assertTrue(exception.message?.contains("Normal distribution only supported for floating point types") == true)
    }

    @Test
    fun testEmptyShape() {
        val shape = Shape()
        val tensorData = factory.full(shape, 0,Int32)
        
        assertEquals(shape, tensorData.shape)
        assertEquals(0, shape.volume)
    }

    @Test
    fun testSingleElementTensor() {
        val shape = Shape(1)
        val tensorData = factory.full(shape,1, FP32)
        
        assertEquals(shape, tensorData.shape)
        assertEquals(1.0f, tensorData[0] as Float)
    }

    @Test
    fun testLargeShape() {
        val shape = Shape(10, 10, 5)
        val tensorData = factory.full(shape, 0, Int32)
        
        assertEquals(shape, tensorData.shape)
        assertEquals(500, shape.volume)
        assertEquals(0, tensorData[9, 9, 4])
    }

    @Test
    fun testIndexOutOfBounds() {
        val shape = Shape(2, 2)
        val tensorData = factory.full(shape, 0, Int32)
        
        assertFailsWith<IllegalArgumentException> {
            tensorData[2, 0] // Should throw bounds exception
        }
    }

    @Test
    fun testWrongNumberOfIndices() {
        val shape = Shape(2, 3)
        val tensorData = factory.full(shape, 0,Int32)
        
        assertFailsWith<IllegalArgumentException> {
            tensorData[0] // Should require 2 indices for 2D tensor
        }
    }

    @Test
    fun testReproducibleRandom() {
        val shape = Shape(3, 3)
        val seed = 42L
        
        val tensorData1 = factory.randn(shape, 0.3f, 2.0f,FP32, Random(seed))
        val tensorData2 = factory.randn(shape, 0.3f, 2.0f, FP32, Random(seed))
        
        // Same seed should produce same results
        assertEquals(tensorData1[0, 0], tensorData2[0, 0])
        assertEquals(tensorData1[2, 2], tensorData2[2, 2])
    }

    @Test
    fun testReproducibleNormal() {
        val shape = Shape(2, 2)
        val seed = 123L
        val mean = 1.5f
        val std = 2.0f
        
        val tensorData1 = factory.randn(shape, mean, std, FP32, Random(seed))
        val tensorData2 = factory.randn(shape, mean, std, FP32, Random(seed))
        
        // Same seed should produce same results
        assertEquals(tensorData1[0, 0], tensorData2[0, 0])
        assertEquals(tensorData1[1, 1], tensorData2[1, 1])
    }
}