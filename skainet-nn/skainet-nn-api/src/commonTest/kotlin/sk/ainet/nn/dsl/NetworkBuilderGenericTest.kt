package sk.ainet.nn.dsl

import sk.ainet.core.tensor.Shape
import sk.ainet.core.tensor.backend.CpuTensorFP32
import sk.ainet.core.tensor.backend.CpuTensorInt8
import sk.ainet.core.tensor.backend.CpuTensorInt32
import sk.ainet.core.tensor.FP32
import sk.ainet.core.tensor.Int8
import sk.ainet.core.tensor.Int32
import sk.ainet.nn.Linear
import sk.ainet.nn.Module
import kotlin.test.*

/**
 * Unit tests for NetworkBuilder generic DSL functionality - Chapter 7.1 requirements
 * Tests different DType combinations, type safety, and backward compatibility
 */
class NetworkBuilderGenericTest {

    @Test
    fun testGenericNetworkFP32() {
        // Test FP32/Float combination - basic functionality
        val network: Module<FP32, Float> = network<FP32, Float> {
            input(2)
            dense(3) {
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
    }

    @Test
    fun testGenericNetworkInt8() {
        // Test Int8/Byte combination
        val network = network<Int8, Byte> {
            input(2)
            dense(2) {
                weights { shape -> CpuTensorInt8.ones(shape) }
                bias { shape -> CpuTensorInt8.zeros(shape) }
            }
        }
        
        assertNotNull(network)
        assertEquals("MLP", network.name)
    }

    @Test
    fun testGenericNetworkInt32() {
        // Test Int32/Int combination
        val network = network<Int32, Int> {
            input(3)
            dense(2) {
                weights { shape -> CpuTensorInt32.ones(shape) }
                bias { shape -> CpuTensorInt32.zeros(shape) }
            }
        }
        
        assertNotNull(network)
        assertEquals("MLP", network.name)
    }

    @Test
    fun testBackwardCompatibilityNonGeneric() {
        // Test that existing non-generic network function still works
        val network = network {
            input(2)
            dense(3) {
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
    }

    @Test
    fun testNetworkFP32HelperFunction() {
        // Test networkFP32 convenience function
        val network = networkFP32 {
            input(2)
            dense(1) {
                weights { shape -> CpuTensorFP32.ones(shape) }
                bias { shape -> CpuTensorFP32.zeros(shape) }
            }
        }
        
        assertNotNull(network)
        assertEquals("MLP", network.name)
    }

    @Test
    fun testFlattenWithDifferentTypes() {
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
        assertEquals("MLP", networkFP32.name)
        assertEquals("MLP", networkInt8.name)
    }

    @Test
    fun testSequentialWithGenericTypes() {
        // Test sequential blocks with generic types
        val network = network<FP32, Float> {
            input(2)
            sequential {
                dense(4) {
                    weights { shape -> CpuTensorFP32.ones(shape) }
                    bias { shape -> CpuTensorFP32.zeros(shape) }
                }
                dense(2) {
                    weights { shape -> CpuTensorFP32.ones(shape) }
                    bias { shape -> CpuTensorFP32.zeros(shape) }
                }
            }
            dense(1) {
                weights { shape -> CpuTensorFP32.ones(shape) }
                bias { shape -> CpuTensorFP32.zeros(shape) }
            }
        }
        
        assertNotNull(network)
        assertEquals("MLP", network.name)
    }

    @Test
    fun testStageWithGenericTypes() {
        // Test stage blocks with generic types
        val network = network<FP32, Float> {
            input(3)
            stage("feature_extraction") {
                dense(8) {
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
        assertEquals("MLP", network.name)
    }

    @Test
    fun testNetworkBuilderClassGeneric() {
        // Test NetworkBuilder class directly with generic types
        val builder = NetworkBuilder<FP32, Float>()
        
        val linear1 = Linear(2, 4, "layer1",
            CpuTensorFP32.ones(Shape(4, 2)), 
            CpuTensorFP32.zeros(Shape(4))
        )
        
        val linear2 = Linear(4, 1, "layer2",
            CpuTensorFP32.ones(Shape(1, 4)),
            CpuTensorFP32.zeros(Shape(1))
        )
        
        val network = builder.add(linear1, linear2).build()
        
        assertNotNull(network)
        assertEquals("MLP", network.name)
    }

    @Test
    fun testTypeSafetyDifferentDTypes() {
        // Test that different DTypes create different network types
        val networkFP32 = network<FP32, Float> {
            input(2)
            dense(1) { 
                weights { CpuTensorFP32.ones(it) }
                bias { CpuTensorFP32.zeros(it) }
            }
        }

        val networkInt8 = network<Int8, Byte> {
            input(2)
            dense(1) { 
                weights { CpuTensorInt8.ones(it) }
                bias { CpuTensorInt8.zeros(it) }
            }
        }

        val networkInt32 = network<Int32, Int> {
            input(2)
            dense(1) { 
                weights { CpuTensorInt32.ones(it) }
                bias { CpuTensorInt32.zeros(it) }
            }
        }
        
        // All should be created successfully
        assertNotNull(networkFP32)
        assertNotNull(networkInt8)
        assertNotNull(networkInt32)
        
        // All should have the same structure name but different implementations
        assertEquals("MLP", networkFP32.name)
        assertEquals("MLP", networkInt8.name)
        assertEquals("MLP", networkInt32.name)
    }
}