package sk.ainet.lang.nn

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Test suite for advanced convolutional operations.
 */
class AdvancedConvolutionsTest {

    private val factory: TensorDataFactory = DenseTensorDataFactory()
    
    private fun createRandomWeights(shape: Shape): Tensor<FP32, Float> {
        val data = factory.randn<FP32, Float>(shape, FP32::class, 0.0f, 0.1f, kotlin.random.Random.Default)
        return VoidOpsTensor(data, FP32::class)
    }
    
    private fun createZeroBias(shape: Shape): Tensor<FP32, Float> {
        val data = factory.zeros<FP32, Float>(shape, FP32::class)
        return VoidOpsTensor(data, FP32::class)
    }

    @Test
    fun testConv2dCreation() {
        val inChannels = 3
        val outChannels = 16
        val kernelSize = 3 to 3
        
        val weightsShape = Shape(intArrayOf(outChannels, inChannels, kernelSize.first, kernelSize.second))
        val biasShape = Shape(intArrayOf(outChannels))
        
        val conv = Conv2d<FP32, Float>(
            inChannels = inChannels,
            outChannels = outChannels,
            kernelSize = kernelSize,
            stride = 1 to 1,
            padding = 1 to 1,
            dilation = 1 to 1,
            groups = 1,
            bias = true,
            name = "test_conv2d",
            initWeights = createRandomWeights(weightsShape),
            initBias = createZeroBias(biasShape)
        )
        
        assertNotNull(conv)
        assertEquals("test_conv2d", conv.name)
        assertEquals(inChannels, conv.inChannels)
        assertEquals(outChannels, conv.outChannels)
        assertEquals(kernelSize, conv.kernelSize)
        assertEquals(2, conv.params.size) // weights + bias
        
        // Test output size calculation
        val inputSize = 32 to 32
        val outputSize = conv.outputSize(inputSize)
        assertEquals(32, outputSize.first) // Same size due to padding=1, stride=1, kernel=3
        assertEquals(32, outputSize.second)
    }

    @Test
    fun testDepthwiseSeparableConv2dCreation() {
        val inChannels = 32
        val outChannels = 64
        val kernelSize = 3 to 3
        
        val depthwiseWeightsShape = Shape(intArrayOf(inChannels, 1, kernelSize.first, kernelSize.second))
        val pointwiseWeightsShape = Shape(intArrayOf(outChannels, inChannels, 1, 1))
        val depthwiseBiasShape = Shape(intArrayOf(inChannels))
        val pointwiseBiasShape = Shape(intArrayOf(outChannels))
        
        val conv = DepthwiseSeparableConv2d<FP32, Float>(
            inChannels = inChannels,
            outChannels = outChannels,
            kernelSize = kernelSize,
            stride = 1 to 1,
            padding = 1 to 1,
            dilation = 1 to 1,
            bias = true,
            name = "test_depthwise_separable",
            initDepthwiseWeights = createRandomWeights(depthwiseWeightsShape),
            initPointwiseWeights = createRandomWeights(pointwiseWeightsShape),
            initDepthwiseBias = createZeroBias(depthwiseBiasShape),
            initPointwiseBias = createZeroBias(pointwiseBiasShape)
        )
        
        assertNotNull(conv)
        assertEquals("test_depthwise_separable", conv.name)
        assertEquals(inChannels, conv.inChannels)
        assertEquals(outChannels, conv.outChannels)
        assertEquals(kernelSize, conv.kernelSize)
        assertEquals(2, conv.modules.size) // depthwise + pointwise modules
        
        // Test parameter count calculation
        val paramCount = conv.parameterCount()
        val expectedParams = (inChannels * kernelSize.first * kernelSize.second + inChannels) +
                           (inChannels * outChannels + outChannels)
        assertEquals(expectedParams.toLong(), paramCount)
        
        // Test parameter reduction
        val reduction = conv.parameterReduction()
        assertTrue(reduction > 1.0, "Depthwise separable should reduce parameters")
    }

    @Test
    fun testGroupedConv2dCreation() {
        val inChannels = 64
        val outChannels = 128
        val kernelSize = 3 to 3
        val groups = 4
        
        val weightsShape = Shape(intArrayOf(outChannels, inChannels / groups, kernelSize.first, kernelSize.second))
        val biasShape = Shape(intArrayOf(outChannels))
        
        val conv = GroupedConv2d<FP32, Float>(
            inChannels = inChannels,
            outChannels = outChannels,
            kernelSize = kernelSize,
            groups = groups,
            stride = 1 to 1,
            padding = 1 to 1,
            dilation = 1 to 1,
            bias = true,
            name = "test_grouped_conv2d",
            initWeights = createRandomWeights(weightsShape),
            initBias = createZeroBias(biasShape)
        )
        
        assertNotNull(conv)
        assertEquals(inChannels, conv.inChannels)
        assertEquals(outChannels, conv.outChannels)
        assertEquals(groups, conv.groups)
        assertEquals(inChannels / groups, conv.inputChannelsPerGroup())
        assertEquals(outChannels / groups, conv.outputChannelsPerGroup())
        
        // Test parameter reduction
        val reduction = conv.parameterReduction()
        assertEquals(groups.toDouble(), reduction, 0.1)
        
        // Test group info
        val info = conv.groupInfo()
        assertEquals(groups, info.groups)
        assertEquals(false, info.isDepthwise)
        assertEquals(false, info.isStandard)
    }

    @Test
    fun testDilatedConv2dCreation() {
        val inChannels = 64
        val outChannels = 128
        val kernelSize = 3 to 3
        val dilation = 2 to 2
        
        val weightsShape = Shape(intArrayOf(outChannels, inChannels, kernelSize.first, kernelSize.second))
        val biasShape = Shape(intArrayOf(outChannels))
        
        val conv = DilatedConv2d<FP32, Float>(
            inChannels = inChannels,
            outChannels = outChannels,
            kernelSize = kernelSize,
            dilation = dilation,
            stride = 1 to 1,
            padding = 2 to 2, // Adjusted for dilation
            groups = 1,
            bias = true,
            name = "test_dilated_conv2d",
            initWeights = createRandomWeights(weightsShape),
            initBias = createZeroBias(biasShape)
        )
        
        assertNotNull(conv)
        assertEquals(dilation, conv.dilation)
        
        // Test effective kernel size calculation
        val effectiveKernel = conv.effectiveKernelSize()
        val expectedSize = kernelSize.first + (kernelSize.first - 1) * (dilation.first - 1)
        assertEquals(expectedSize, effectiveKernel.first)
        assertEquals(expectedSize, effectiveKernel.second)
        
        // Test receptive field expansion
        val expansion = conv.receptiveFieldExpansion()
        assertTrue(expansion > 1.0, "Dilated convolution should expand receptive field")
        
        // Test dilation info
        val info = conv.dilationInfo()
        assertEquals(dilation, info.dilation)
        assertEquals(false, info.isStandard)
    }

    @Test
    fun testTransposedConv2dCreation() {
        val inChannels = 128
        val outChannels = 64
        val kernelSize = 4 to 4
        val stride = 2 to 2
        
        val weightsShape = Shape(intArrayOf(inChannels, outChannels, kernelSize.first, kernelSize.second))
        val biasShape = Shape(intArrayOf(outChannels))
        
        val conv = TransposedConv2d<FP32, Float>(
            inChannels = inChannels,
            outChannels = outChannels,
            kernelSize = kernelSize,
            stride = stride,
            padding = 1 to 1,
            outputPadding = 0 to 0,
            dilation = 1 to 1,
            groups = 1,
            bias = true,
            name = "test_transposed_conv2d",
            initWeights = createRandomWeights(weightsShape),
            initBias = createZeroBias(biasShape)
        )
        
        assertNotNull(conv)
        assertEquals(stride, conv.stride)
        assertEquals(true, conv.is2xUpsampling())
        
        // Test output size calculation (upsampling)
        val inputSize = 16 to 16
        val outputSize = conv.outputSize(inputSize)
        assertEquals(32, outputSize.first) // 2x upsampling
        assertEquals(32, outputSize.second)
        
        // Test upsampling factor
        val factor = conv.upsamplingFactor()
        assertEquals(2.0, factor.first)
        assertEquals(2.0, factor.second)
        
        // Test transposed conv info
        val info = conv.transposedConvInfo()
        assertEquals(true, info.is2xUpsampling)
        assertEquals(true, info.isStandard)
    }

    @Test
    fun testTransposedConv2dFactoryMethods() {
        val inChannels = 64
        val outChannels = 32
        val weightsShape = Shape(intArrayOf(inChannels, outChannels, 4, 4))
        val biasShape = Shape(intArrayOf(outChannels))
        
        // Test 2x upsampling factory
        val conv2x = TransposedConv2d.upsampling2x<FP32, Float>(
            inChannels = inChannels,
            outChannels = outChannels,
            initWeights = createRandomWeights(weightsShape),
            initBias = createZeroBias(biasShape)
        )
        
        assertEquals(2 to 2, conv2x.stride)
        assertTrue(conv2x.is2xUpsampling())
        
        // Test custom upsampling factory
        val customWeightsShape = Shape(intArrayOf(inChannels, outChannels, 6, 6))
        val conv4x = TransposedConv2d.customUpsampling<FP32, Float>(
            inChannels = inChannels,
            outChannels = outChannels,
            upsamplingFactor = 3 to 3,
            initWeights = createRandomWeights(customWeightsShape)
        )
        
        assertEquals(3 to 3, conv4x.stride)
        assertEquals(3.0, conv4x.upsamplingFactor().first)
    }
}