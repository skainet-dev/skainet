package sk.ainet.nn.dsl

import sk.ainet.core.tensor.Shape
import sk.ainet.core.tensor.backend.CpuTensorFP32
import sk.ainet.core.tensor.FP32
import sk.ainet.core.tensor.Int8
import sk.ainet.core.tensor.Int32
import sk.ainet.core.tensor.backend.CpuBackend
import sk.ainet.core.tensor.backend.CpuBackendInt8
import sk.ainet.core.tensor.backend.CpuBackendInt32
import kotlin.test.*
import kotlin.random.Random

/**
 * Additional test coverage for NetworkBuilder DSL focusing on uncovered functionality.
 * This test class complements existing tests to improve overall package coverage.
 */
class NetworkBuilderCoverageTest {

    @Test
    fun testNetworkWithAutoFactoryErrorHandling() {
        // Test unsupported type combination - should throw IllegalArgumentException
        assertFailsWith<IllegalArgumentException> {
            network<FP32, Int> {
                input(10)
                dense(5)
            }
        }
    }

    @Test
    fun testDenseLayerWithoutOutputDimension() {
        // Test dense layer configuration without specifying output dimension
        val net = network<FP32, Float>(CpuBackend()) {
            input(10)
            dense {
                units = 5
                weights { ones(it) }
                bias { zeros(it) }
            }
        }
        assertNotNull(net)
    }

    @Test
    fun testAdvancedRandomInitializationMethods() {
        // Test various random initialization methods not covered in existing tests
        val customRandom = Random(seed = 12345)
        val net = network<FP32, Float>(CpuBackend()) {
            input(8)
            dense(6) {
                weights { randomNormal(it, mean = 0.0, std = 0.1, customRandom) }
                bias { randomUniform(it, min = -0.05, max = 0.05, seed = 999L) }
            }
            dense(4) {
                weights { randomUniform(it, min = -0.1, max = 0.1, customRandom) }
                bias { randomNormal(it, mean = 0.0, std = 0.01, seed = 42L) }
            }
        }
        assertNotNull(net)
    }

    @Test
    fun testComplexNestedNetworkStructure() {
        // Test complex nested structures with stages and sequential blocks
        val net = network<FP32, Float>(CpuBackend()) {
            input(784)
            stage("encoder") {
                dense(256) {
                    weights { randomNormal(it, 0.0, 0.02) }
                    bias { zeros(it) }
                }
                activation("encoder_activation") { tensor -> tensor }
                sequential {
                    dense(128) {
                        weights { randomUniform(it, -0.1, 0.1) }
                        bias { zeros(it) }
                    }
                    flatten("encoder_flatten")
                }
            }
            stage("classifier") {
                dense(64) {
                    weights { ones(it) }
                    bias { randomNormal(it, 0.0, 0.01) }
                }
                dense(10) {
                    weights { randomNormal(it, 0.0, 0.05) }
                    bias { zeros(it) }
                }
            }
        }
        assertNotNull(net)
    }

    @Test
    fun testFlattenLayerWithCustomConfiguration() {
        // Test flatten layer with custom start and end dimensions
        val net = network<FP32, Float>(CpuBackend()) {
            input(28 * 28)
            flatten("custom_flatten") {
                startDim = 1
                endDim = -1
            }
            dense(128) {
                weights { randomNormal(it, 0.0, 0.1) }
                bias { zeros(it) }
            }
            dense(10) {
                weights { ones(it) }
                bias { zeros(it) }
            }
        }
        assertNotNull(net)
    }

    @Test
    fun testMultipleActivationLayers() {
        // Test multiple activation layers with different functions
        val net = network<FP32, Float>(CpuBackend()) {
            input(20)
            dense(15) { weights { ones(it) } }
            activation("first_activation") { tensor -> tensor }
            dense(10) { weights { randomNormal(it, 0.0, 0.1) } }
            activation("second_activation") { tensor ->
                // Custom activation for testing
                tensor
            }
            dense(5) { weights { randomUniform(it, -0.1, 0.1) } }
        }
        assertNotNull(net)
    }

    @Test
    fun testNetworkBuilderClassDirectUsage() {
        // Test NetworkBuilder class direct usage
        val builder = NetworkBuilder<FP32, Float>()
        
        val linear1 = sk.ainet.nn.Linear(3, 5, "layer1", 
            CpuTensorFP32.ones(Shape(5, 3)), 
            CpuTensorFP32.zeros(Shape(1, 5))
        )
        
        val linear2 = sk.ainet.nn.Linear(5, 2, "layer2",
            CpuTensorFP32.ones(Shape(2, 5)),
            CpuTensorFP32.zeros(Shape(1, 2))
        )
        
        val network = builder.add(linear1, linear2).build()
        assertNotNull(network)
        assertEquals("MLP", network.name)
    }

    @Test
    fun testInt8NetworkWithAdvancedFeatures() {
        // Test Int8 networks with various initialization methods
        val net = network<Int8, Byte>(CpuBackendInt8()) {
            input(16)
            dense(8) {
                weights { randomNormal(it, 0.0, 1.0, seed = 123L) }
                bias { zeros(it) }
            }
            sequential {
                dense(4) {
                    weights { randomUniform(it, -2.0, 2.0) }
                    bias { ones(it) }
                }
                flatten("int8_flatten")
            }
            dense(2) {
                weights { random(it, seed = 456L) }
                bias { randomNormal(it, 0.0, 0.5) }
            }
        }
        assertNotNull(net)
    }

    @Test
    fun testInt32NetworkWithComplexStructure() {
        // Test Int32 networks with stages and different initialization
        val customRandom = Random(seed = 789)
        val net = network<Int32, Int>(CpuBackendInt32()) {
            input(12)
            stage("processing") {
                dense(8) {
                    weights { randomNormal(it, 0.0, 1.0, customRandom) }
                    bias { randomUniform(it, -1.0, 1.0, customRandom) }
                }
                activation("int32_activation") { tensor -> tensor }
            }
            stage("output") {
                dense(4) {
                    weights { random(it, customRandom) }
                    bias { zeros(it) }
                }
            }
        }
        assertNotNull(net)
    }

    @Test
    fun testEmptyAndMinimalConfigurations() {
        // Test minimal network configurations
        val net1 = network<FP32, Float>(CpuBackend()) {
            input(1)
        }
        assertNotNull(net1)
        
        val net2 = network<FP32, Float>(CpuBackend()) {
            input(2)
            dense(1) { weights { ones(it) } }
        }
        assertNotNull(net2)
    }

    @Test
    fun testMixedInitializationStrategies() {
        // Test mixing different initialization strategies
        val net = network<FP32, Float>(CpuBackend()) {
            input(6)
            dense(12) {
                weights { shape ->
                    // Conditional initialization based on shape
                    if (shape[0] > 10) randomNormal(shape, 0.0, 0.1) else ones(shape)
                }
                bias { randomUniform(it, -0.01, 0.01) }
            }
            dense(8) {
                weights { randomNormal(it, 0.0, 0.05, seed = 555L) }
                bias { shape -> CpuTensorFP32.full(shape, 0.1f) }
            }
            dense(3) {
                weights { ones(it) }
                bias { zeros(it) }
            }
        }
        assertNotNull(net)
    }

    @Test
    fun testSequentialAndStageNesting() {
        // Test various combinations of sequential and stage nesting
        val net = network<FP32, Float>(CpuBackend()) {
            input(10)
            sequential {
                stage("nested_stage") {
                    dense(8) {
                        weights { randomNormal(it, 0.0, 0.1) }
                        bias { zeros(it) }
                    }
                    sequential {
                        dense(6) {
                            weights { randomUniform(it, -0.1, 0.1) }
                            bias { zeros(it) }
                        }
                        activation("nested_activation") { tensor -> tensor }
                    }
                }
            }
            dense(4) {
                weights { ones(it) }
                bias { zeros(it) }
            }
        }
        assertNotNull(net)
    }

    @Test
    fun testDifferentRandomSeedsAndInstances() {
        // Test different random seed and instance combinations
        val random1 = Random(seed = 111)
        val random2 = Random(seed = 222)
        
        val net = network<FP32, Float>(CpuBackend()) {
            input(5)
            dense(4) {
                weights { random(it, random1) }
                bias { randomNormal(it, 0.0, 0.1, random2) }
            }
            dense(3) {
                weights { randomUniform(it, -0.5, 0.5, seed = 333L) }
                bias { randomNormal(it, 0.0, 0.05, seed = 444L) }
            }
            dense(2) {
                weights { randomUniform(it, -0.1, 0.1, random1) }
                bias { random(it, seed = 555L) }
            }
        }
        assertNotNull(net)
    }

    @Test
    fun testEdgeCasesAndBoundaryConditions() {
        // Test various edge cases and boundary conditions
        val net = network<FP32, Float>(CpuBackend()) {
            input(1)  // Single input
            dense(1) {  // Single output
                weights { ones(it) }
                bias { zeros(it) }
            }
        }
        assertNotNull(net)
        
        // Test with larger dimensions
        val net2 = network<FP32, Float>(CpuBackend()) {
            input(1000)
            dense(500) {
                weights { randomNormal(it, 0.0, 0.01) }
                bias { zeros(it) }
            }
            dense(100) {
                weights { randomUniform(it, -0.001, 0.001) }
                bias { zeros(it) }
            }
            dense(10) {
                weights { ones(it) }
                bias { zeros(it) }
            }
        }
        assertNotNull(net2)
    }

    @Test
    fun testBackwardCompatibilityFunctions() {
        // Test backward compatibility functions
        val net1 = networkFP32 {
            input(5)
            dense(3) {
                weights { ones(it) }
                bias { zeros(it) }
            }
        }
        assertNotNull(net1)
        
        val net2 = network {
            input(4)
            dense(2) {
                weights { randomNormal(it, 0.0, 0.1) }
                bias { zeros(it) }
            }
        }
        assertNotNull(net2)
    }
}