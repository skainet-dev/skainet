package sk.ainet.lang.tensor.dsl

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.testFactory
import sk.ainet.lang.types.FP32
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

    @Test
    fun testBasicTensorCreation() {
        val dtype = FP32

        // Test basic tensor creation with zeros
        val zerosTensor = tensor<FP32, Float>(testFactory) {
            shape(3, 4) { zeros() }
        }

        assertNotNull(zerosTensor)
        assertEquals(Shape(3, 4), zerosTensor.shape)
        assertEquals(0.0f, zerosTensor.data[0, 0])

        // Test basic tensor creation with ones
        val onesTensor = tensor<FP32, Float>(testFactory) {
            shape(2, 3) { ones() }
        }

        assertNotNull(onesTensor)
        assertEquals(Shape(2, 3), onesTensor.shape)
        assertEquals(1.0f, onesTensor.data[0, 0])
    }

    @Test
    fun testMLPWeightCreation() {
        val dtype = FP32

        // Test Xavier initialization for MLP weights
        val xavierWeights = tensor<FP32, Float>(testFactory) {
            shape(64, 128) {
                val limit = kotlin.math.sqrt(6.0f / (128 + 64))
                uniform(-limit, limit) 
            }
        }

        assertNotNull(xavierWeights)
        assertEquals(Shape(64, 128), xavierWeights.shape)

        // Test He initialization for ReLU networks
        val heWeights = tensor<FP32, Float>(testFactory) {
            shape(128, 256) {
                val std = kotlin.math.sqrt(2.0f / 256)
                randn(mean = 0.0f, std = std) 
            }
        }

        assertNotNull(heWeights)
        assertEquals(Shape(128, 256), heWeights.shape)

        // Test normal weight initialization
        val normalWeights = tensor<FP32, Float>(testFactory) {
            shape(50, 100) { randn(mean = 0.0f, std = 0.02f) }
        }

        assertNotNull(normalWeights)
        assertEquals(Shape(50, 100), normalWeights.shape)
    }

    @Test
    fun testMLPBiasCreation() {
        val dtype = FP32

        // Test zero bias initialization (most common)
        val zeroBias = tensor<FP32, Float>(testFactory) {
            shape(64) { zeros() }
        }

        assertNotNull(zeroBias)
        assertEquals(Shape(64), zeroBias.shape)
        assertEquals(0.0f, zeroBias.data[0])

        // Test constant bias initialization
        val constantBias = tensor<FP32, Float>(testFactory) {
            shape(32) { full(0.1f) }
        }

        assertNotNull(constantBias)
        assertEquals(Shape(32), constantBias.shape)
    }

    @Test
    fun testImageTensorCreation() {
        val dtype = FP32

        // Test BCHW format for image batches
        val imageBatch = tensor<FP32, Float>(testFactory) {
            shape(8, 3, 224, 224) { zeros() }
        }

        assertNotNull(imageBatch)
        assertEquals(Shape(8, 3, 224, 224), imageBatch.shape)

        // Test single RGB image
        val rgbImage = tensor<FP32, Float>(testFactory) {
            shape(3, 64, 64) { uniform(0.0f, 1.0f) }
        }

        assertNotNull(rgbImage)
        assertEquals(Shape(3, 64, 64), rgbImage.shape)

        // Test grayscale image
        val grayscaleImage = tensor<FP32, Float>(testFactory) {
            shape(1, 128, 128) { zeros() }
        }

        assertNotNull(grayscaleImage)
        assertEquals(Shape(1, 128, 128), grayscaleImage.shape)
    }

    @Test
    fun testConvolutionKernelCreation() {
        val dtype = FP32

        // Test 3x3 convolution kernel
        val conv3x3 = tensor<FP32, Float>(testFactory) {
            shape(64, 32, 3, 3) { randn(mean = 0.0f, std = 0.1f) }
        }

        assertNotNull(conv3x3)
        assertEquals(Shape(64, 32, 3, 3), conv3x3.shape)

        // Test pointwise (1x1) convolution
        val pointwise = tensor<FP32, Float>(testFactory) {
            shape(128, 64, 1, 1) {
                val limit = kotlin.math.sqrt(6.0f / (64 + 128))
                uniform(-limit, limit) 
            }
        }

        assertNotNull(pointwise)
        assertEquals(Shape(128, 64, 1, 1), pointwise.shape)
    }

    @Test
    fun testCustomInitialization() {
        val dtype = FP32

        // Test custom initialization with index-based function
        val customTensor = tensor<FP32, Float>(testFactory) {
            shape(3, 3) {
                init { indices ->
                    (indices[0] + indices[1]).toFloat() // Sum of row and column indices
                }
            }
        }

        assertNotNull(customTensor)
        assertEquals(Shape(3, 3), customTensor.shape)
        assertEquals(0.0f, customTensor.data[0, 0]) // 0 + 0
        assertEquals(2.0f, customTensor.data[1, 1]) // 1 + 1

        // Test custom random initialization
        val customRandom = tensor<FP32, Float>(testFactory) {
            shape(2, 2) {
                randomInit({ random ->
                    if (random.nextBoolean()) 1.0f else -1.0f // Random sign
                })
            }
        }

        assertNotNull(customRandom)
        assertEquals(Shape(2, 2), customRandom.shape)
        // Values should be either 1.0f or -1.0f
        assertTrue(customRandom.data[0, 0] == 1.0f || customRandom.data[0, 0] == -1.0f)
    }

    @Test
    fun testDistributionInitialization() {
        val dtype = FP32

        // Test normal distribution
        val normalTensor = tensor<FP32, Float>(testFactory) {
            shape(100) { randn(mean = 5.0f, std = 2.0f) }
        }

        assertNotNull(normalTensor)
        assertEquals(Shape(100), normalTensor.shape)

        // Test uniform distribution
        val uniformTensor = tensor<FP32, Float>(testFactory) {
            shape(50) { uniform(min = -1.0f, max = 1.0f) }
        }

        assertNotNull(uniformTensor)
        assertEquals(Shape(50), uniformTensor.shape)
    }

    @Test
    fun testImageInitializers() {
        val dtype = FP32

        // Test normalized pixel initialization
        val normalizedImage = tensor<FP32, Float>(testFactory) {
            shape(3, 32, 32) { uniform(0.0f, 1.0f) }
        }

        assertNotNull(normalizedImage)
        assertEquals(Shape(3, 32, 32), normalizedImage.shape)

        // Test black image
        val blackImage = tensor<FP32, Float>(testFactory) {
            shape(1, 64, 64) { zeros() }
        }

        assertNotNull(blackImage)
        assertEquals(Shape(1, 64, 64), blackImage.shape)
        assertEquals(0.0f, blackImage.data[0, 0, 0])
    }
}