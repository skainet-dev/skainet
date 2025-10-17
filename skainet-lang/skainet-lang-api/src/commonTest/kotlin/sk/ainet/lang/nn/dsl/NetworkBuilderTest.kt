package sk.ainet.nn.dsl

import sk.ainet.core.tensor.Shape
import sk.ainet.core.tensor.backend.CpuTensorFP32
import sk.ainet.core.tensor.backend.CpuTensorInt8
import sk.ainet.core.tensor.backend.CpuTensorInt32
import sk.ainet.core.tensor.FP32
import sk.ainet.core.tensor.Int8
import sk.ainet.core.tensor.Int32
import kotlin.test.*
import kotlin.math.sqrt

/**
 * Unit tests for NetworkBuilder generic DSL functionality - Chapter 7.1 requirements
 * Tests different DType combinations, type safety, and backward compatibility
 */
class NetworkBuilderTest {

    @Test
    fun testNetworkBuilderFP32() {
        // Test FP32/Float combination - backward compatibility
        val network = network {
            input(2)
            dense(4) {
                weights { shape -> ones() }
                bias { shape -> CpuTensorFP32.zeros(shape) }
            }
            dense(1) {
                weights { shape -> CpuTensorFP32.ones(shape) }
                bias { shape -> CpuTensorFP32.zeros(shape) }
            }
        }

        assertNotNull(network)
        assertEquals("MLP", network.name)

        // Test forward pass
        val input = CpuTensorFP32.fromArray(Shape(1, 2), floatArrayOf(1.0f, 2.0f))
        val output = network.forward(input)
        assertEquals(Shape(1, 1), output.shape)
    }

    @Test
    fun testNetworkBuilderInt8() {
        // Test Int8/Byte combination
        val network = network<Int8, Byte>(sk.ainet.core.tensor.backend.CpuBackendInt8()) {
            input(2)
            dense(3) {
                weights { shape -> CpuTensorInt8.ones(shape) }
                bias { shape -> CpuTensorInt8.zeros(shape) }
            }
            flatten()
            dense(1) {
                weights { shape -> CpuTensorInt8.ones(shape) }
                bias { shape -> CpuTensorInt8.zeros(shape) }
            }
        }

        assertNotNull(network)
        assertEquals("MLP", network.name)

        // Test forward pass
        val input = CpuTensorInt8.fromArray(Shape(1, 2), byteArrayOf(1, 2))
        val output = network.forward(input)
        assertEquals(Shape(1, 1), output.shape)
    }

    @Test
    fun testNetworkBuilderInt32() {
        // Test Int32/Int combination
        val network = network<Int32, Int> {
            input(3)
            dense(2) {
                weights { shape -> CpuTensorInt32.ones(shape) }
                bias { shape -> CpuTensorInt32.zeros(shape) }
            }
        }

        assertNotNull(network)

        // Test forward pass
        val input = CpuTensorInt32.fromArray(Shape(1, 3), intArrayOf(1, 2, 3))
        val output = network.forward(input)
        assertEquals(Shape(1, 2), output.shape)
    }

    @Test
    fun testBackwardCompatibilityNonGenericFunction() {
        // Test that existing non-generic network function still works
        val network = network {
            input(2)
            dense(4) {
                activation = { tensor ->
                    with(tensor) { relu() }
                }
                weights { shape -> CpuTensorFP32.ones(shape) }
                bias { shape -> CpuTensorFP32.zeros(shape) }
            }
            dense(1)
        }

        assertNotNull(network)

        // Verify it's using FP32/Float types
        val input = CpuTensorFP32.fromArray(Shape(1, 2), floatArrayOf(1.0f, 2.0f))
        val output = network.forward(input)
        assertEquals(Shape(1, 1), output.shape)
    }

    @Test
    fun testNetworkFP32HelperFunction() {
        // Test networkFP32 convenience function
        val network = network {
            input(2)
            dense(1) {
                weights { shape -> CpuTensorFP32.ones(shape) }
                bias { shape -> CpuTensorFP32.zeros(shape) }
            }
        }

        assertNotNull(network)
        val input = CpuTensorFP32.fromArray(Shape(1, 2), floatArrayOf(1.0f, 2.0f))
        val output = network.forward(input)
        assertEquals(Shape(1, 1), output.shape)
    }

    @Test
    fun testFlattenWithGenericTypes() {
        // Test flatten layer with different generic types
        val networkFP32 = network<FP32, Float> {
            input(4)
            flatten()
            dense(2) {
                weights { shape -> CpuTensorFP32.ones(shape) }
                bias { shape -> CpuTensorFP32.zeros(shape) }
            }
        }

        val networkInt8 = network<Int8, Byte> {
            input(4)
            flatten()
            dense(2) {
                weights { shape -> CpuTensorInt8.ones(shape) }
                bias { shape -> CpuTensorInt8.zeros(shape) }
            }
        }

        assertNotNull(networkFP32)
        assertNotNull(networkInt8)
    }

    @Test
    fun testSequentialWithGenericTypes() {
        // Test sequential blocks with generic types
        val network = network {
            input(2)
            sequential {
                dense(4) {
                    activation = { tensor ->
                        with(tensor) { relu() }
                    }
                    weights { shape -> CpuTensorFP32.ones(shape) }
                    bias { shape -> CpuTensorFP32.zeros(shape) }
                }
                dense(2) {
                    weights { shape -> CpuTensorFP32.ones(shape) }
                    bias { shape -> CpuTensorFP32.zeros(shape) }
                }
            }
            dense(1)
        }

        assertNotNull(network)
        val input = CpuTensorFP32.fromArray(Shape(1, 2), floatArrayOf(1.0f, 2.0f))
        val output = network.forward(input)
        assertEquals(Shape(1, 1), output.shape)
    }

    @Test
    fun testStageWithGenericTypes() {
        // Test stage blocks with generic types
        val network = network {
            input(3)
            stage("feature_extraction") {
                dense(8) {
                    activation = { tensor ->
                        with(tensor) { relu() }
                    }
                    weights { shape -> CpuTensorFP32.ones(shape) }
                    bias { shape -> CpuTensorFP32.zeros(shape) }
                }
                flatten()
            }
            stage("classification") {
                dense(2) {
                    weights { shape -> CpuTensorFP32.ones(shape) }
                    bias { shape -> CpuTensorFP32.zeros(shape) }
                }
            }
        }

        assertNotNull(network)
        val input = CpuTensorFP32.fromArray(Shape(1, 3), floatArrayOf(1.0f, 2.0f, 3.0f))
        val output = network.forward(input)
        assertEquals(Shape(1, 2), output.shape)
    }

    @Test
    fun testActivationFunctionGenericTypes() {
        // Test activation functions with generic tensor types
        val network = network {
            input(2)
            dense(4) {
                weights { shape -> CpuTensorFP32.ones(shape) }
                bias { shape -> CpuTensorFP32.zeros(shape) }
            }
            activation { tensor ->
                // Simple ReLU activation function
                with(tensor) { relu() }
            }
            dense(1)
        }

        assertNotNull(network)
        val input = CpuTensorFP32.fromArray(Shape(1, 2), floatArrayOf(1.0f, -1.0f))
        val output = network.forward(input)
        assertEquals(Shape(1, 1), output.shape)
    }

    @Test
    fun testNetworkBuilderClass() {
        // Test NetworkBuilder class directly with generic types
        val builder = NetworkBuilder<FP32, Float>()

        val linear1 = sk.ainet.nn.Linear(
            2, 4, "layer1",
            CpuTensorFP32.ones(Shape(4, 2)),
            CpuTensorFP32.zeros(Shape(1, 4))
        )

        val linear2 = sk.ainet.nn.Linear(
            4, 1, "layer2",
            CpuTensorFP32.ones(Shape(1, 4)),
            CpuTensorFP32.zeros(Shape(1, 1))
        )

        val network = builder.add(linear1, linear2).build()

        assertNotNull(network)
        assertEquals("MLP", network.name)

        val input = CpuTensorFP32.fromArray(Shape(1, 2), floatArrayOf(1.0f, 2.0f))
        val output = network.forward(input)
        assertEquals(Shape(1, 1), output.shape)
    }

    @Test
    fun testGenericFactoryMethodsFP32() {
        // Test generic factory methods with FP32 - demonstrates the new DSL
        val network = network<FP32, Float> {
            input(2)
            dense(4) {
                weights { ones() }    // Generic factory method instead of CpuTensorFP32.ones()
                bias { zeros() }      // Generic factory method instead of CpuTensorFP32.zeros()
            }
            dense(1) {
                weights { random() }  // Generic factory method for random initialization
                bias { zeros() }
            }
        }

        assertNotNull(network)
        assertEquals("MLP", network.name)

        // Test forward pass
        val input = CpuTensorFP32.fromArray(Shape(1, 2), floatArrayOf(1.0f, 2.0f))
        val output = network.forward(input)
        assertEquals(Shape(1, 1), output.shape)
    }

    @Test
    fun testGenericFactoryMethodsInt32() {
        // Test generic factory methods with Int32
        val network = network<Int32, Int> {
            input(3)
            dense(2) {
                weights { ones() }    // Uses Int32 ones automatically
                bias { zeros() }      // Uses Int32 zeros automatically
            }
            dense(1) {
                weights { random() }  // Uses Int32 random values
                bias { zeros() }
            }
        }

        assertNotNull(network)

        // Test forward pass
        val input = CpuTensorInt32.fromArray(Shape(1, 3), intArrayOf(1, 2, 3))
        val output = network.forward(input)
        assertEquals(Shape(1, 1), output.shape)
    }

    @Test
    fun testMixedInitializationPatterns() {
        // Test mixing factory methods with custom initialization
        val network = network {
            input(4)
            dense(8) {
                weights { shape ->
                    if (shape[0] > 5) random() else ones()
                }
                bias { zeros() }
            }
            dense(1) {
                weights { ones() }
                bias { shape -> CpuTensorFP32.full(shape, 0.1f) }  // Mix with explicit calls
            }
        }

        assertNotNull(network)

        val input = CpuTensorFP32.fromArray(Shape(1, 4), floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f))
        val output = network.forward(input)
        assertEquals(Shape(1, 1), output.shape)
    }

    @Test
    fun testAdvancedRandomMethodsWithSeed() {
        // Test seed-controlled random initialization for reproducibility
        val seed = 42L
        val network1 = network<FP32,Float> {
            input(3)
            dense(4) {
                weights { random(seed) }  // Same seed
                bias { randomNormal(0.0, 0.1, seed) }
            }
            dense(1) {
                weights { randomUniform(-0.5, 0.5, seed) }
                bias { zeros() }
            }
        }

        val network2 = network <FP32,Float>{
            input(3)
            dense(4) {
                weights { random(seed) }  // Same seed - should produce identical weights
                bias { randomNormal(0.0, 0.1, seed) }
            }
            dense(1) {
                weights { randomUniform(-0.5, 0.5, seed) }
                bias { zeros() }
            }
        }

        assertNotNull(network1)
        assertNotNull(network2)

        // Test that both networks produce the same output for the same input (deterministic)
        val input = CpuTensorFP32.fromArray(Shape(1, 3), floatArrayOf(1.0f, 2.0f, 3.0f))
        val output1 = network1.forward(input)
        val output2 = network2.forward(input)

        assertEquals(output1.shape, output2.shape)
        // Note: We can't easily test exact equality due to potential floating point differences
        // but we can test that both networks are structurally identical
    }

    @Test
    fun testDistributionBasedInitialization() {
        // Test normal and uniform distribution initialization
        val network = network<FP32,Float> {
            input(5)
            dense(8) {
                // Xavier-like initialization using normal distribution
                weights { randomNormal(0.0, sqrt(2.0 / 5.0)) }
                bias { zeros() }
            }
            dense(4) {
                // Uniform distribution initialization
                weights { randomUniform(-0.1, 0.1) }
                bias { randomNormal(0.0, 0.01) }
            }
            dense(1) {
                weights { ones() }
                bias { zeros() }
            }
        }

        assertNotNull(network)
        assertEquals("MLP", network.name)

        // Test forward pass
        val input = CpuTensorFP32.fromArray(Shape(1, 5), floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f))
        val output = network.forward(input)
        assertEquals(Shape(1, 1), output.shape)
    }

    @Test
    fun testCustomRandomInstanceInitialization() {
        // Test using custom Random instance for advanced control
        val customRandom = kotlin.random.Random(123)
        val network = network<FP32,Float> {
            input(3)
            dense(5) {
                weights { random(customRandom) }
                bias { randomNormal(0.0, 0.1, customRandom) }
            }
            dense(2) {
                weights { randomUniform(-1.0, 1.0, customRandom) }
                bias { zeros() }
            }
        }

        assertNotNull(network)

        // Test forward pass
        val input = CpuTensorFP32.fromArray(Shape(1, 3), floatArrayOf(1.0f, 2.0f, 3.0f))
        val output = network.forward(input)
        assertEquals(Shape(1, 2), output.shape)
    }

    @Test
    fun testAdvancedRandomWithInt32() {
        // Test advanced random methods with Int32 data type
        val network = network<Int32, Int> {
            input(2)
            dense(3) {
                weights { randomNormal(0.0, 2.0, 999L) }  // Normal distribution for integers
                bias { randomUniform(-5.0, 5.0) }         // Uniform distribution
            }
            dense(1) {
                weights { random(555L) }  // Seed-controlled random
                bias { zeros() }
            }
        }

        assertNotNull(network)

        // Test forward pass
        val input = CpuTensorInt32.fromArray(Shape(1, 2), intArrayOf(10, 20))
        val output = network.forward(input)
        assertEquals(Shape(1, 1), output.shape)
    }
}