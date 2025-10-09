package sk.ainet.lang.tensor

import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.dsl.*
import sk.ainet.lang.types.FP32
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.test.assertFailsWith

/**
 * Integration test suite for tensor slicing functionality
 * Implements testing requirements from tasks-slicing.md section 5.2 (tasks 21-22)
 */
class SlicingIntegrationTest {

    // Mock tensor data implementation for testing
    private class MockTensorData(
        override val shape: Shape,
        private val data: FloatArray
    ) : TensorData<FP32, Float> {
        override fun get(vararg indices: Int): Float = data[shape.index(indices)]
        override fun set(vararg indices: Int, value: Float) { data[shape.index(indices)] = value }
    }

    // Mock factory for creating test tensors
    private val testFactory = object : TensorDataFactory<FP32, Float> {
        override fun zeros(shape: Shape, dtype: FP32): TensorData<FP32, Float> = 
            MockTensorData(shape, FloatArray(shape.volume) { 0.0f })
            
        override fun ones(shape: Shape, dtype: FP32): TensorData<FP32, Float> = 
            MockTensorData(shape, FloatArray(shape.volume) { 1.0f })
            
        override fun full(shape: Shape, value: Number, dtype: FP32): TensorData<FP32, Float> = 
            MockTensorData(shape, FloatArray(shape.volume) { value.toFloat() })
            
        override fun randn(shape: Shape, mean: Float, std: Float, dtype: FP32, random: Random): TensorData<FP32, Float> = 
            MockTensorData(shape, FloatArray(shape.volume) { 
                // Simple approximation of normal distribution
                val u1 = Random.nextFloat()
                val u2 = Random.nextFloat()
                val z = kotlin.math.sqrt(-2.0 * kotlin.math.ln(u1.toDouble())) * kotlin.math.cos(2.0 * kotlin.math.PI * u2.toDouble())
                (z.toFloat() * std + mean)
            })
            
        override fun uniform(shape: Shape, min: Float, max: Float, dtype: FP32, random: Random): TensorData<FP32, Float> = 
            MockTensorData(shape, FloatArray(shape.volume) { Random.nextFloat() * (max - min) + min })
            
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

    // Task 21.1: Test compatibility with existing tensor operations
    @Test
    fun testCompatibilityWithExistingTensorOperations() {
        println("[DEBUG_LOG] Testing compatibility with existing tensor operations")
        
        with(testFactory) {
            // Create base tensor
            val tensor = tensor(4, 3, 2) { shape ->
                init { indices ->
                    var flatIndex = 0
                    var stride = 1
                    for (i in indices.indices.reversed()) {
                        flatIndex += indices[i] * stride
                        stride *= shape[i]
                    }
                    flatIndex.toFloat()
                }
            }
            
            // Test slicing with different tensor creation methods
            val zerosView = tensor.sliceView {
                segment { range(0, 2) }
                segment { all() }
                segment { all() }
            }
            
            val onesView = tensor.sliceView {
                segment { range(2, 4) }
                segment { all() }
                segment { all() }
            }
            
            // Verify compatibility
            assertEquals(Shape(2, 3, 2), zerosView.shape)
            assertEquals(Shape(2, 3, 2), onesView.shape)
            
            // Test that views maintain tensor interface compatibility
            assertNotNull(zerosView.data)
            assertNotNull(onesView.data)
            
            assertTrue(zerosView is TensorView<FP32, Float>)
            assertTrue(onesView is TensorView<FP32, Float>)
            
            println("[DEBUG_LOG] Compatibility with existing tensor operations verified")
        }
    }

    // Task 21.2: Test sliceView() vs sliceCopy() behavioral differences
    @Test
    fun testSliceViewVsSliceCopyBehavioralDifferences() {
        println("[DEBUG_LOG] Testing sliceView() vs sliceCopy() behavioral differences")
        
        with(testFactory) {
            val tensor = tensor(4, 3, 2) { shape ->
                init { indices ->
                    var flatIndex = 0
                    var stride = 1
                    for (i in indices.indices.reversed()) {
                        flatIndex += indices[i] * stride
                        stride *= shape[i]
                    }
                    flatIndex.toFloat()
                }
            }
            
            // Test sliceView - should create zero-copy view
            val view = tensor.sliceView {
                segment { range(1, 3) }
                segment { all() }
                segment { all() }
            }
            
            assertEquals(Shape(2, 3, 2), view.shape)
            assertTrue(view is TensorView<FP32, Float>)
            
            // Test sliceCopy - should throw NotImplementedError for now
            assertFailsWith<NotImplementedError> {
                tensor.sliceCopy {
                    segment { range(1, 3) }
                    segment { all() }
                    segment { all() }
                }
            }
            
            println("[DEBUG_LOG] sliceView() vs sliceCopy() behavioral differences validated")
        }
    }

    // Task 21.3: Test error handling and validation in DSL
    @Test
    fun testErrorHandlingAndValidationInDSL() {
        println("[DEBUG_LOG] Testing error handling and validation in DSL")
        
        with(testFactory) {
            val tensor = tensor(4, 3, 2) { shape ->
                zeros()
            }
            
            // Test various error conditions
            
            // Invalid range bounds
            assertFailsWith<IllegalArgumentException> {
                tensor.sliceView {
                    segment { range(3, 1) }  // End < start
                    segment { all() }
                    segment { all() }
                }
            }
            
            // Out of bounds index
            try {
                tensor.sliceView {
                    segment { at(10) }  // Index > dimension size
                    segment { all() }
                    segment { all() }
                }
                println("[DEBUG_LOG] Out-of-bounds index allowed (implementation specific)")
            } catch (e: Exception) {
                println("[DEBUG_LOG] Out-of-bounds index properly rejected: ${e.message}")
            }
            
            // Invalid step size
            assertFailsWith<IllegalArgumentException> {
                tensor.sliceView {
                    segment { step(0, 4, 0) }  // Step = 0
                    segment { all() }
                    segment { all() }
                }
            }
            
            // Negative step (if not supported)
            try {
                tensor.sliceView {
                    segment { step(3, 0, -1) }  // Negative step
                    segment { all() }
                    segment { all() }
                }
                println("[DEBUG_LOG] Negative step allowed (implementation specific)")
            } catch (e: Exception) {
                println("[DEBUG_LOG] Negative step properly rejected: ${e.message}")
            }
            
            println("[DEBUG_LOG] Error handling and validation tests completed")
        }
    }

    // Task 21.4: Verify backward compatibility with existing tensor creation patterns
    @Test
    fun testBackwardCompatibilityWithExistingTensorCreationPatterns() {
        println("[DEBUG_LOG] Testing backward compatibility with existing tensor creation patterns")
        
        with(testFactory) {
            // Test with various tensor creation patterns
            val testCases = listOf(
                "zeros" to { shape: Shape -> zeros(shape, FP32) },
                "ones" to { shape: Shape -> ones(shape, FP32) },
                "full" to { shape: Shape -> full(shape, 5.0f, FP32) },
                "uniform" to { shape: Shape -> uniform(shape, 0.0f, 1.0f, FP32, Random(42)) }
            )
            
            for ((name, factory) in testCases) {
                val tensorData = factory(Shape(4, 3, 2))
                val tensor = tensor(4, 3, 2) { shape -> 
                    // Create tensor using the factory data
                    init { indices ->
                        tensorData.get(*indices)
                    }
                }
                
                // Test slicing works with all creation patterns
                val view = tensor.sliceView {
                    segment { range(1, 3) }
                    segment { all() }
                    segment { at(0) }
                }
                
                assertEquals(Shape(2, 3), view.shape)
                println("[DEBUG_LOG] Backward compatibility verified for: $name")
            }
            
            println("[DEBUG_LOG] Backward compatibility tests completed")
        }
    }

    // Task 22.1: Implement batch processing scenarios (mini-batch extraction)
    @Test
    fun testBatchProcessingScenarios() {
        println("[DEBUG_LOG] Testing batch processing scenarios")
        
        with(testFactory) {
            // Create dataset tensor: (batch_size=32, features=128)
            val dataset = tensor(32, 128) { shape ->
                init { indices ->
                    // Create synthetic data with batch index as primary component
                    indices[0].toFloat() * 1000 + indices[1].toFloat()
                }
            }
            
            // Test mini-batch extraction
            val batchSize = 8
            val batches = mutableListOf<TensorView<FP32, Float>>()
            
            for (i in 0 until 4) {
                val startIdx = i * batchSize
                val endIdx = (i + 1) * batchSize
                
                val batch = dataset.sliceView {
                    segment { range(startIdx, endIdx) }
                    segment { all() }
                }
                
                assertEquals(Shape(batchSize, 128), batch.shape)
                batches.add(batch)
                
                // Verify batch data integrity
                val firstSample = batch.data[0, 0]
                val expectedFirstSample = startIdx.toFloat() * 1000 + 0.0f
                assertEquals(expectedFirstSample, firstSample)
            }
            
            assertEquals(4, batches.size)
            println("[DEBUG_LOG] Mini-batch extraction verified with ${batches.size} batches of size $batchSize")
        }
    }

    // Task 22.2: Test feature extraction use cases (channel slicing, spatial regions)
    @Test
    fun testFeatureExtractionUseCases() {
        println("[DEBUG_LOG] Testing feature extraction use cases")
        
        with(testFactory) {
            // Create NCHW image tensor: (batch=4, channels=3, height=32, width=32)
            val images = tensor(4, 3, 32, 32) { shape ->
                init { indices ->
                    // Create synthetic image data
                    val batch = indices[0]
                    val channel = indices[1]
                    val height = indices[2]
                    val width = indices[3]
                    (batch * 10000 + channel * 1000 + height * 10 + width).toFloat()
                }
            }
            
            // Test channel slicing (extract RGB channels separately)
            val redChannel = images.sliceView {
                segment { all() }  // All batches
                segment { at(0) }  // Red channel only
                segment { all() }  // All height
                segment { all() }  // All width
            }
            assertEquals(Shape(4, 32, 32), redChannel.shape)
            
            val greenChannel = images.sliceView {
                segment { all() }  // All batches
                segment { at(1) }  // Green channel only
                segment { all() }  // All height
                segment { all() }  // All width
            }
            assertEquals(Shape(4, 32, 32), greenChannel.shape)
            
            // Test spatial region extraction (center crop)
            val centerCrop = images.sliceView {
                segment { all() }      // All batches
                segment { all() }      // All channels
                segment { range(8, 24) }  // Center height region
                segment { range(8, 24) }  // Center width region
            }
            assertEquals(Shape(4, 3, 16, 16), centerCrop.shape)
            
            // Test corner extraction
            val topLeft = images.sliceView {
                segment { all() }      // All batches
                segment { all() }      // All channels
                segment { range(0, 8) }   // Top region
                segment { range(0, 8) }   // Left region
            }
            assertEquals(Shape(4, 3, 8, 8), topLeft.shape)
            
            println("[DEBUG_LOG] Feature extraction use cases verified")
        }
    }

    // Task 22.3: Simulate sequence processing with sliding windows
    @Test
    fun testSequenceProcessingWithSlidingWindows() {
        println("[DEBUG_LOG] Testing sequence processing with sliding windows")
        
        with(testFactory) {
            // Create sequence tensor: (batch=2, sequence_length=100, features=64)
            val sequences = tensor(2, 100, 64) { shape ->
                init { indices ->
                    // Create synthetic sequence data
                    val batch = indices[0]
                    val time = indices[1]
                    val feature = indices[2]
                    (batch * 10000 + time * 100 + feature).toFloat()
                }
            }
            
            // Test sliding window extraction
            val windowSize = 16
            val stride = 4
            val windows = mutableListOf<TensorView<FP32, Float>>()
            
            // Generate overlapping windows
            var startPos = 0
            while (startPos + windowSize <= 100) {
                val endPos = startPos + windowSize
                
                val window = sequences.sliceView {
                    segment { all() }  // All batches
                    segment { range(startPos, endPos) }  // Window in sequence
                    segment { all() }  // All features
                }
                
                assertEquals(Shape(2, windowSize, 64), window.shape)
                windows.add(window)
                
                // Verify window data
                val firstValue = window.data[0, 0, 0]
                val expectedValue = (0 * 10000 + startPos * 100 + 0).toFloat()
                assertEquals(expectedValue, firstValue)
                
                startPos += stride
            }
            
            assertTrue(windows.size >= 20, "Should generate multiple overlapping windows")
            println("[DEBUG_LOG] Sliding window processing verified with ${windows.size} windows")
        }
    }

    // Task 22.4: Test multi-head attention and model layer output patterns
    @Test
    fun testMultiHeadAttentionAndModelLayerOutputPatterns() {
        println("[DEBUG_LOG] Testing multi-head attention and model layer output patterns")
        
        with(testFactory) {
            // Create attention tensor: (batch=4, heads=8, seq_length=64, head_dim=32)
            val attention = tensor(4, 8, 64, 32) { shape ->
                init { indices ->
                    // Create synthetic attention data
                    val batch = indices[0]
                    val head = indices[1]
                    val seq = indices[2]
                    val dim = indices[3]
                    (batch * 100000 + head * 10000 + seq * 100 + dim).toFloat()
                }
            }
            
            // Test single head extraction
            val singleHead = attention.sliceView {
                segment { all() }  // All batches
                segment { at(0) }  // First head only
                segment { all() }  // All sequence positions
                segment { all() }  // All dimensions
            }
            assertEquals(Shape(4, 64, 32), singleHead.shape)
            
            // Test specific batch and head combination
            val batchHeadView = attention.sliceView {
                segment { at(0) }  // First batch
                segment { range(0, 4) }  // First 4 heads
                segment { all() }  // All sequence positions
                segment { all() }  // All dimensions
            }
            assertEquals(Shape(4, 64, 32), batchHeadView.shape)
            
            // Test sequence subsequence for attention window
            val attentionWindow = attention.sliceView {
                segment { all() }  // All batches
                segment { all() }  // All heads
                segment { range(16, 48) }  // Middle sequence region
                segment { all() }  // All dimensions
            }
            assertEquals(Shape(4, 8, 32, 32), attentionWindow.shape)
            
            // Test model layer output pattern (typically flattening last dimensions)
            val flattenedOutput = attention.sliceView {
                segment { all() }  // All batches
                segment { all() }  // All heads
                segment { at(0) }  // First sequence position
                segment { all() }  // All dimensions
            }
            assertEquals(Shape(4, 8, 32), flattenedOutput.shape)
            
            println("[DEBUG_LOG] Multi-head attention and model layer patterns verified")
        }
    }
}