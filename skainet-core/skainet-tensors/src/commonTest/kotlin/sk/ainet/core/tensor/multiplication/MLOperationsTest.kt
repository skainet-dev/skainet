package sk.ainet.core.tensor.multiplication

import sk.ainet.core.tensor.*
import sk.ainet.core.tensor.backend.CpuBackend
import sk.ainet.core.tensor.backend.CpuTensorFP32
import kotlin.test.*

class MLOperationsTest {

    private val backend = CpuBackend()

    @Test
    fun testLinearLayerForward() {
        // Simulate a linear layer: input * weights + bias
        // Input: batch_size=2, features=3
        // Weights: features=3, output_features=4
        val inputShape = Shape(2, 3)  // 2 samples, 3 features each
        val weightShape = Shape(3, 4) // 3 input features, 4 output features

        val input = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f) // 2x3 input batch
        val weights = FloatArray(12) { (it + 1).toFloat() } // 3x4 weight matrix

        val inputTensor = CpuTensorFP32.fromArray(inputShape, input)
        val weightTensor = CpuTensorFP32.fromArray(weightShape, weights)

        val result = backend.matmul(inputTensor, weightTensor)

        assertEquals(Shape(2, 4), result.shape) // 2 samples, 4 output features

        // First sample output: [1*1 + 2*5 + 3*9, 1*2 + 2*6 + 3*10, 1*3 + 2*7 + 3*11, 1*4 + 2*8 + 3*12]
        // = [38, 44, 50, 56]
        assertEquals(38f, result[0, 0])
        assertEquals(44f, result[0, 1])
        assertEquals(50f, result[0, 2])
        assertEquals(56f, result[0, 3])

        // Second sample output: [4*1 + 5*5 + 6*9, 4*2 + 5*6 + 6*10, 4*3 + 5*7 + 6*11, 4*4 + 5*8 + 6*12]
        // = [83, 98, 113, 128]
        assertEquals(83f, result[1, 0])
        assertEquals(98f, result[1, 1])
        assertEquals(113f, result[1, 2])
        assertEquals(128f, result[1, 3])
    }

    @Test
    fun testConvolutionalWeightMultiplication() {
        // Simulate flattened convolution operation
        // Feature map: NCHW format - 1 batch, 2 channels, 3x3 spatial
        val featureShape = Shape(1, 2, 3, 3) // 1x2x3x3
        val filterShape = Shape(18, 8) // Flattened: 2*3*3=18 input features, 8 filters

        val featureData = FloatArray(18) { (it + 1).toFloat() }
        val filterData = FloatArray(144) { (it % 5 + 1).toFloat() }

        // Flatten feature map for matrix multiplication
        val flatFeatureShape = Shape(1, 18) // Batch size 1, 18 features
        val flatFeature = CpuTensorFP32.fromArray(flatFeatureShape, featureData)
        val filters = CpuTensorFP32.fromArray(filterShape, filterData)

        val result = backend.matmul(flatFeature, filters)

        assertEquals(Shape(1, 8), result.shape) // 1 batch, 8 output channels
        assertTrue(result[0, 0] > 0f)
        assertTrue(result[0, 7] > 0f)
    }

    @Test
    fun testBatchedLinearTransformation() {
        // Batch processing: multiple samples through the same linear layer
        val batchSize = 5
        val inputFeatures = 10
        val outputFeatures = 7

        val inputShape = Shape(batchSize, inputFeatures)
        val weightShape = Shape(inputFeatures, outputFeatures)

        val inputData = FloatArray(50) { (it % 3 + 1).toFloat() }
        val weightData = FloatArray(70) { (it % 4 + 1).toFloat() }

        val input = CpuTensorFP32.fromArray(inputShape, inputData)
        val weights = CpuTensorFP32.fromArray(weightShape, weightData)

        val result = backend.matmul(input, weights)

        assertEquals(Shape(batchSize, outputFeatures), result.shape)

        // Each sample should produce different outputs
        assertNotEquals(result[0, 0], result[1, 0])
        assertNotEquals(result[0, 0], result[2, 0])
    }

    @Test
    fun testAttentionWeightMultiplication() {
        // Simulate attention mechanism: Q * K^T
        val sequenceLength = 8
        val embeddingDim = 16

        val queryShape = Shape(sequenceLength, embeddingDim) // 8x16
        val keyShape = Shape(embeddingDim, sequenceLength)   // 16x8 (transposed)

        val queryData = FloatArray(128) { (it % 7 + 1).toFloat() }
        val keyData = FloatArray(128) { (it % 5 + 1).toFloat() }

        val query = CpuTensorFP32.fromArray(queryShape, queryData)
        val key = CpuTensorFP32.fromArray(keyShape, keyData)

        val attentionScores = backend.matmul(query, key)

        assertEquals(Shape(sequenceLength, sequenceLength), attentionScores.shape) // 8x8

        // Attention scores should be symmetric in terms of dimensions
        assertTrue(attentionScores[0, 0] > 0f)
        assertTrue(attentionScores[7, 7] > 0f)
    }

    @Test
    fun testEmbeddingLookup() {
        // Simulate embedding lookup as matrix multiplication with one-hot vectors
        val vocabSize = 1000
        val embeddingDim = 128
        val batchSize = 3

        // One-hot encoded input (simplified - only test a few positions)
        val oneHotShape = Shape(batchSize, vocabSize)
        val embeddingShape = Shape(vocabSize, embeddingDim)

        // Create sparse one-hot vectors (mostly zeros with single 1s)
        val oneHotData = FloatArray(3000) { 0f }
        oneHotData[0] = 1f    // First sample: word at position 0
        oneHotData[1001] = 1f // Second sample: word at position 1
        oneHotData[2002] = 1f // Third sample: word at position 2

        val embeddingData = FloatArray(128000) { (it % 10 + 1).toFloat() }

        val oneHot = CpuTensorFP32.fromArray(oneHotShape, oneHotData)
        val embeddings = CpuTensorFP32.fromArray(embeddingShape, embeddingData)

        val result = backend.matmul(oneHot, embeddings)

        assertEquals(Shape(batchSize, embeddingDim), result.shape)

        // Each sample should get different embeddings
        assertNotEquals(result[0, 0], result[1, 0])
        assertNotEquals(result[1, 0], result[2, 0])
    }

    @Test
    fun testMultiHeadAttentionProjection() {
        // Multiple attention heads processing in parallel
        val batchSize = 2
        val sequenceLength = 6
        val embeddingDim = 12
        val numHeads = 3
        val headDim = embeddingDim / numHeads // 4

        val inputShape = Shape(batchSize * sequenceLength, embeddingDim) // 12x12
        val projectionShape = Shape(embeddingDim, embeddingDim) // 12x12

        val inputData = FloatArray(144) { (it % 8 + 1).toFloat() }
        val projectionData = FloatArray(144) { (it % 6 + 1).toFloat() }

        val input = CpuTensorFP32.fromArray(inputShape, inputData)
        val projection = CpuTensorFP32.fromArray(projectionShape, projectionData)

        val projected = backend.matmul(input, projection)

        assertEquals(Shape(batchSize * sequenceLength, embeddingDim), projected.shape)
        assertTrue(projected[0, 0] > 0f)
        assertTrue(projected[11, 11] > 0f)
    }

    @Test
    fun testResidualConnectionWeights() {
        // Test weight matrices used in residual connections
        val hiddenSize = 256
        val intermediateSize = 1024
        val batchSize = 4

        // Forward: hidden -> intermediate
        val inputShape = Shape(batchSize, hiddenSize)
        val forwardWeightShape = Shape(hiddenSize, intermediateSize)

        val input = FloatArray(1024) { (it % 12 + 1).toFloat() }
        val forwardWeights = FloatArray(262144) { (it % 7 + 1).toFloat() }

        val inputTensor = CpuTensorFP32.fromArray(inputShape, input)
        val forwardTensor = CpuTensorFP32.fromArray(forwardWeightShape, forwardWeights)

        val intermediate = backend.matmul(inputTensor, forwardTensor)

        assertEquals(Shape(batchSize, intermediateSize), intermediate.shape)

        // Backward: intermediate -> hidden
        val backwardWeightShape = Shape(intermediateSize, hiddenSize)
        val backwardWeights = FloatArray(262144) { (it % 9 + 1).toFloat() }
        val backwardTensor = CpuTensorFP32.fromArray(backwardWeightShape, backwardWeights)

        val output = backend.matmul(intermediate, backwardTensor)

        assertEquals(Shape(batchSize, hiddenSize), output.shape)
        assertTrue(output[0, 0] > 0f)
        assertTrue(output[3, 255] > 0f)
    }

    @Test
    fun testGradientAccumulation() {
        // Simulate gradient computation through matrix multiplication
        val batchSize = 8
        val features = 64
        val outputs = 32

        val activationShape = Shape(batchSize, features)
        val gradientShape = Shape(batchSize, outputs)
        val weightGradientShape = Shape(features, outputs)

        val activations = FloatArray(512) { (it % 5 + 1).toFloat() }
        val outputGradients = FloatArray(256) { (it % 3 + 1).toFloat() }

        val activationTensor = CpuTensorFP32.fromArray(activationShape, activations)
        val gradientTensor = CpuTensorFP32.fromArray(gradientShape, outputGradients)

        // Transpose activations for gradient computation: A^T * grad_output
        val transposeShape = Shape(features, batchSize)
        val transposedActivations = FloatArray(512)

        // Manual transpose for testing
        for (i in 0 until batchSize) {
            for (j in 0 until features) {
                transposedActivations[j * batchSize + i] = activations[i * features + j]
            }
        }

        val transposedTensor = CpuTensorFP32.fromArray(transposeShape, transposedActivations)
        val weightGradients = backend.matmul(transposedTensor, gradientTensor)

        assertEquals(weightGradientShape, weightGradients.shape)
        assertTrue(weightGradients[0, 0] > 0f)
        assertTrue(weightGradients[63, 31] > 0f)
    }

    @Test
    fun testBatchNormalizationStatistics() {
        // Compute batch statistics using matrix operations
        val batchSize = 16
        val channels = 64
        val spatialSize = 49 // 7x7

        // Reshape from NCHW to matrix for batch norm computation
        val inputShape = Shape(batchSize, channels * spatialSize)
        val statisticsWeightShape = Shape(channels * spatialSize, channels)

        val inputData = FloatArray(batchSize * channels * spatialSize) { (it % 11 + 1).toFloat() }
        val weightData = FloatArray(channels * spatialSize * channels) { if (it % (spatialSize + 1) == 0) 1f else 0f }

        val input = CpuTensorFP32.fromArray(inputShape, inputData)
        val weights = CpuTensorFP32.fromArray(statisticsWeightShape, weightData)

        val channelSums = backend.matmul(input, weights)

        assertEquals(Shape(batchSize, channels), channelSums.shape)
        assertTrue(channelSums[0, 0] > 0f)
    }

    @Test
    @Ignore
    fun testLargeScaleMLMultiplication() {
        // Test large matrix multiplication similar to modern ML models
        val batchSize = 32
        val sequenceLength = 512
        val hiddenSize = 768

        // Typical transformer layer sizes
        val inputShape = Shape(batchSize, sequenceLength * hiddenSize / 16) // Reduced for testing
        val weightShape = Shape(sequenceLength * hiddenSize / 16, hiddenSize / 4)

        val inputData = FloatArray(inputShape.volume) { (it % 13 + 1).toFloat() }
        val weightData = FloatArray(weightShape.volume) { (it % 17 + 1).toFloat() }

        val input = CpuTensorFP32.fromArray(inputShape, inputData)
        val weights = CpuTensorFP32.fromArray(weightShape, weightData)

        val result = backend.matmul(input, weights)

        assertEquals(Shape(batchSize, hiddenSize / 4), result.shape)
        assertTrue(result[0, 0] > 0f)
        assertTrue(result[31, 191] > 0f)
    }

    @Test
    fun testMLPWithTensorList() {
        // Test MLP with 1 input, 2 hidden layers (16 neurons each), 1 output using list of tensors
        val batchSize = 4
        val inputSize = 1
        val hiddenSize = 16
        val outputSize = 1

        // Create input data (4 samples, 1 feature each)
        val inputData = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
        val input = CpuTensorFP32.fromArray(Shape(batchSize, inputSize), inputData)

        // Create list of weight tensors for MLP layers
        val weights = mutableListOf<CpuTensorFP32>()
        val biases = mutableListOf<CpuTensorFP32>()

        // Layer 1: input_size -> hidden_size (1 -> 16)
        val w1Data = FloatArray(inputSize * hiddenSize) { (it % 5 + 1).toFloat() / 10f } // Small weights
        val b1Data = FloatArray(batchSize * hiddenSize) { 0.1f } // Bias for each batch
        weights.add(CpuTensorFP32.fromArray(Shape(inputSize, hiddenSize), w1Data))
        biases.add(CpuTensorFP32.fromArray(Shape(batchSize, hiddenSize), b1Data))

        // Layer 2: hidden_size -> hidden_size (16 -> 16)  
        val w2Data = FloatArray(hiddenSize * hiddenSize) { (it % 3 + 1).toFloat() / 20f } // Small weights
        val b2Data = FloatArray(batchSize * hiddenSize) { 0.05f }
        weights.add(CpuTensorFP32.fromArray(Shape(hiddenSize, hiddenSize), w2Data))
        biases.add(CpuTensorFP32.fromArray(Shape(batchSize, hiddenSize), b2Data))

        // Layer 3: hidden_size -> output_size (16 -> 1)
        val w3Data = FloatArray(hiddenSize * outputSize) { (it % 7 + 1).toFloat() / 30f } // Small weights
        val b3Data = FloatArray(batchSize * outputSize) { 0.0f }
        weights.add(CpuTensorFP32.fromArray(Shape(hiddenSize, outputSize), w3Data))
        biases.add(CpuTensorFP32.fromArray(Shape(batchSize, outputSize), b3Data))

        // Forward pass through MLP
        var activation: Tensor<FP32, Float> = input

        // Layer 1: input -> hidden1
        activation = backend.matmul(activation, weights[0])
        // Add bias using tensor addition
        with(backend) {
            activation = activation.plus(biases[0])
        }

        // Apply ReLU activation (max(0, x))
        val relu1Data = FloatArray(activation.shape.volume)
        for (i in relu1Data.indices) {
            relu1Data[i] = maxOf(0f, (activation as CpuTensorFP32).data[i])
        }
        activation = CpuTensorFP32.fromArray(activation.shape, relu1Data)

        // Layer 2: hidden1 -> hidden2
        activation = backend.matmul(activation, weights[1])
        // Add bias using tensor addition
        with(backend) {
            activation = activation.plus(biases[1])
        }

        // Apply ReLU activation
        val relu2Data = FloatArray(activation.shape.volume)
        for (i in relu2Data.indices) {
            relu2Data[i] = maxOf(0f, (activation as CpuTensorFP32).data[i])
        }
        activation = CpuTensorFP32.fromArray(activation.shape, relu2Data)

        // Layer 3: hidden2 -> output
        val output = backend.matmul(activation, weights[2])

        // Verify MLP structure
        assertEquals(3, weights.size, "Should have 3 weight matrices")
        assertEquals(3, biases.size, "Should have 3 bias vectors")

        // Verify layer shapes
        assertEquals(Shape(inputSize, hiddenSize), weights[0].shape, "Layer 1 weight shape")
        assertEquals(Shape(hiddenSize, hiddenSize), weights[1].shape, "Layer 2 weight shape")
        assertEquals(Shape(hiddenSize, outputSize), weights[2].shape, "Layer 3 weight shape")

        // Verify output shape
        assertEquals(Shape(batchSize, outputSize), output.shape, "Output should be batch_size x output_size")

        // Verify output values are reasonable (not NaN or infinite)
        for (i in 0 until batchSize) {
            val outputValue = output[i, 0]
            assertTrue(outputValue.isFinite(), "Output should be finite")
            println("[DEBUG_LOG] Sample $i: input=${inputData[i]}, output=$outputValue")
        }

        // Verify different inputs produce different outputs
        assertNotEquals(output[0, 0], output[1, 0], "Different inputs should produce different outputs")

        println("[DEBUG_LOG] MLP test completed successfully with ${weights.size} layers")
    }
}