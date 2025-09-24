package sk.ainet.nn.dsl

import sk.ainet.core.tensor.Shape
import sk.ainet.core.tensor.backend.CpuTensorFP32
import sk.ainet.core.tensor.backend.CpuTensorInt8
import sk.ainet.core.tensor.backend.CpuTensorInt32
import sk.ainet.core.tensor.FP32
import sk.ainet.core.tensor.Int8
import sk.ainet.core.tensor.Int32
import sk.ainet.core.tensor.Tensor
import sk.ainet.core.tensor.TensorFactory
import kotlin.test.*

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
                weights { shape -> CpuTensorFP32.ones(shape) }
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
        val output = with(network) { input.forward(input) }
        assertEquals(Shape(1, 1), output.shape)
    }

    @Test
    fun testNetworkBuilderInt8() {
        // Test Int8/Byte combination
        val network = network<Int8, Byte>(object:TensorFactory<Int8, Byte>{
            override fun zeros(shape: Shape): Tensor<Int8, Byte> {
                TODO("Not yet implemented")
            }

            override fun ones(shape: Shape): Tensor<Int8, Byte> {
                TODO("Not yet implemented")
            }

        }) {
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
        val output = with(network) { input.forward(input) }
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
        val output = with(network) { input.forward(input) }
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
        val output = with(network) { input.forward(input) }
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
        val output = with(network) { input.forward(input) }
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
        val output = with(network) { input.forward(input) }
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
        val output = with(network) { input.forward(input) }
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
        val output = with(network) { input.forward(input) }
        assertEquals(Shape(1, 1), output.shape)
    }

    @Test
    fun testNetworkBuilderClass() {
        // Test NetworkBuilder class directly with generic types
        val builder = NetworkBuilder<FP32, Float>()
        
        val linear1 = sk.ainet.nn.Linear(2, 4, "layer1", 
            CpuTensorFP32.ones(Shape(4, 2)), 
            CpuTensorFP32.zeros(Shape(1, 4))
        )
        
        val linear2 = sk.ainet.nn.Linear(4, 1, "layer2",
            CpuTensorFP32.ones(Shape(1, 4)),
            CpuTensorFP32.zeros(Shape(1, 1))
        )
        
        val network = builder.add(linear1, linear2).build()
        
        assertNotNull(network)
        assertEquals("MLP", network.name)
        
        val input = CpuTensorFP32.fromArray(Shape(1, 2), floatArrayOf(1.0f, 2.0f))
        val output = with(network) { input.forward(input) }
        assertEquals(Shape(1, 1), output.shape)
    }
}