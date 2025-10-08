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
 * Unit tests demonstrating the usage of the Tensor DSL for creating tensors
 * 
 * This test suite shows how to use the DSL for:
 * - MLP (weights, bias) tensor creation
 * - Image processing (BCHW) tensor creation
 * - Various initializers: zeros, ones, random distributions
 * - Custom initialization functions
 */
class TensorDSLTest {
    
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
                // Simple normal distribution approximation using Box-Muller transform
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
    
    // Mock TensorData for testing
    private class MockTensorData(
        override val shape: Shape,
        private val data: FloatArray
    ) : TensorData<FP32, Float> {
        override fun get(vararg indices: Int): Float = data[shape.index(indices)]
        override fun set(vararg indices: Int, value: Float) { data[shape.index(indices)] = value }
    }

    @Test
    fun testBasicTensorCreation() {
        val dtype = FP32()
        
        // Test basic tensor creation with zeros
        val zerosTensor = tensor(dtype)
            .shape(3, 4)
            .zeros()
            .build(testFactory)
            
        assertNotNull(zerosTensor)
        assertEquals(Shape(3, 4), zerosTensor.shape)
        assertEquals(0.0f, zerosTensor.data[0, 0])
        
        // Test basic tensor creation with ones
        val onesTensor = tensor(dtype)
            .shape(2, 3)
            .ones()
            .build(testFactory)
            
        assertNotNull(onesTensor)
        assertEquals(Shape(2, 3), onesTensor.shape)
        assertEquals(1.0f, onesTensor.data[0, 0])
    }

    @Test
    fun testMLPWeightCreation() {
        val dtype = FP32()
        
        // Test Xavier initialization for MLP weights
        val xavierWeights = tensor<FP32, Float>(dtype)
            .weights()
            .xavier(inputSize = 128, outputSize = 64)
            .build(testFactory)
            
        assertNotNull(xavierWeights)
        assertEquals(Shape(64, 128), xavierWeights.shape)
        
        // Test He initialization for ReLU networks
        val heWeights = tensor<FP32, Float>(dtype)
            .weights()
            .he(inputSize = 256, outputSize = 128)
            .build(testFactory)
            
        assertNotNull(heWeights)
        assertEquals(Shape(128, 256), heWeights.shape)
        
        // Test normal weight initialization
        val normalWeights = tensor<FP32, Float>(dtype)
            .weights()
            .normal(inputSize = 100, outputSize = 50, mean = 0.0f, std = 0.02f)
            .build(testFactory)
            
        assertNotNull(normalWeights)
        assertEquals(Shape(50, 100), normalWeights.shape)
    }

    @Test
    fun testMLPBiasCreation() {
        val dtype = FP32()
        
        // Test zero bias initialization (most common)
        val zeroBias = tensor<FP32, Float>(dtype)
            .bias()
            .zeros(64)
            .build(testFactory)
            
        assertNotNull(zeroBias)
        assertEquals(Shape(64), zeroBias.shape)
        assertEquals(0.0f, zeroBias.data[0])
        
        // Test constant bias initialization
        val constantBias = tensor<FP32, Float>(dtype)
            .bias()
            .constant(32, 0.1f)
            .build(testFactory)
            
        assertNotNull(constantBias)
        assertEquals(Shape(32), constantBias.shape)
    }

    @Test
    fun testImageTensorCreation() {
        val dtype = FP32()
        
        // Test BCHW format for image batches
        val imageBatch = tensor<FP32, Float>(dtype)
            .image()
            .bchw(batchSize = 8, channels = 3, height = 224, width = 224)
            .zeros()
            .build(testFactory)
            
        assertNotNull(imageBatch)
        assertEquals(Shape(8, 3, 224, 224), imageBatch.shape)
        
        // Test single RGB image
        val rgbImage = tensor<FP32, Float>(dtype)
            .image()
            .rgb(height = 64, width = 64)
            .uniform(0.0f, 1.0f)
            .build(testFactory)
            
        assertNotNull(rgbImage)
        assertEquals(Shape(3, 64, 64), rgbImage.shape)
        
        // Test grayscale image
        val grayscaleImage = tensor<FP32, Float>(dtype)
            .image()
            .grayscale(height = 128, width = 128)
            .zeros()
            .build(testFactory)
            
        assertNotNull(grayscaleImage)
        assertEquals(Shape(1, 128, 128), grayscaleImage.shape)
    }

    @Test
    fun testConvolutionKernelCreation() {
        val dtype = FP32()
        
        // Test 3x3 convolution kernel
        val conv3x3 = tensor<FP32, Float>(dtype)
            .kernel()
            .conv3x3(outputChannels = 64, inputChannels = 32)
            .randn(mean = 0.0f, std = 0.1f)
            .build(testFactory)
            
        assertNotNull(conv3x3)
        assertEquals(Shape(64, 32, 3, 3), conv3x3.shape)
        
        // Test pointwise (1x1) convolution
        val pointwise = tensor<FP32, Float>(dtype)
            .kernel()
            .pointwise(outputChannels = 128, inputChannels = 64)
            .xavier(64, 128)
            .build(testFactory)
            
        assertNotNull(pointwise)
        assertEquals(Shape(128, 64, 1, 1), pointwise.shape)
    }

    @Test
    fun testCustomInitialization() {
        val dtype = FP32()
        
        // Test custom initialization with index-based function
        val customTensor = tensor<FP32, Float>(dtype)
            .shape(3, 3)
            .init { indices -> 
                (indices[0] + indices[1]).toFloat() // Sum of row and column indices
            }
            .build(testFactory)
            
        assertNotNull(customTensor)
        assertEquals(Shape(3, 3), customTensor.shape)
        assertEquals(0.0f, customTensor.data[0, 0]) // 0 + 0
        assertEquals(2.0f, customTensor.data[1, 1]) // 1 + 1
        
        // Test custom random initialization
        val customRandom = tensor<FP32, Float>(dtype)
            .shape(2, 2)
            .randomInit { random -> 
                if (random.nextBoolean()) 1.0f else -1.0f // Random sign
            }
            .build(testFactory)
            
        assertNotNull(customRandom)
        assertEquals(Shape(2, 2), customRandom.shape)
        // Values should be either 1.0f or -1.0f
        assertTrue(customRandom.data[0, 0] == 1.0f || customRandom.data[0, 0] == -1.0f)
    }

    @Test
    fun testDistributionInitialization() {
        val dtype = FP32()
        
        // Test normal distribution
        val normalTensor = tensor<FP32, Float>(dtype)
            .shape(100)
            .randn(mean = 5.0f, std = 2.0f)
            .build(testFactory)
            
        assertNotNull(normalTensor)
        assertEquals(Shape(100), normalTensor.shape)
        
        // Test uniform distribution
        val uniformTensor = tensor<FP32, Float>(dtype)
            .shape(50)
            .uniform(min = -1.0f, max = 1.0f)
            .build(testFactory)
            
        assertNotNull(uniformTensor)
        assertEquals(Shape(50), uniformTensor.shape)
    }

    @Test
    fun testImageInitializers() {
        val dtype = FP32()
        
        // Test normalized pixel initialization
        val normalizedImage = tensor<FP32, Float>(dtype)
            .image()
            .rgb(32, 32)
            .imageInit()
            .normalizedPixels()
            .build(testFactory)
            
        assertNotNull(normalizedImage)
        assertEquals(Shape(3, 32, 32), normalizedImage.shape)
        
        // Test black image
        val blackImage = tensor<FP32, Float>(dtype)
            .image()
            .grayscale(64, 64)
            .imageInit()
            .black()
            .build(testFactory)
            
        assertNotNull(blackImage)
        assertEquals(Shape(1, 64, 64), blackImage.shape)
        assertEquals(0.0f, blackImage.data[0, 0, 0])
    }
}