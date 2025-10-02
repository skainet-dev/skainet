package sk.ainet.nn.dsl

import sk.ainet.core.tensor.Shape
import sk.ainet.core.tensor.FP32
import sk.ainet.nn.Linear
import sk.ainet.nn.topology.ModuleParameters
import sk.ainet.nn.topology.weights
import sk.ainet.nn.topology.bias
import sk.ainet.nn.dsl.extensions.*
import sk.ainet.core.tensor.factory.fromBytes
import sk.ainet.core.tensor.factory.ByteArrayConverter
import kotlin.test.*

/**
 * Unit tests for SinNetwork DSL compilation and shape validation.
 * Tests that the network compiles successfully and all weights and biases have correct shapes.
 */
class SinNetworkShapeTest {

    @Test
    fun testSinNetworkCompilationAndShapes() {
        println("[DEBUG_LOG] Starting SinNetwork compilation and shape validation test")
        
        // Build the network using standard DSL syntax
        val sinNetwork = network<FP32, Float> {
            input(1)  // Single input for x value
            
            // First hidden layer: 1 -> 16 neurons
            dense(16) {
                // Weights: 16x1 matrix - explicitly defined values
                weights {
                    fromArray(
                        0.5f, -0.3f, 0.8f, -0.2f, 0.6f, -0.4f, 0.7f, -0.1f,
                        0.9f, -0.5f, 0.3f, -0.7f, 0.4f, -0.6f, 0.2f, -0.8f
                    )
                }
                
                // Bias: 16 values - explicitly defined
                bias {
                    fromArray(
                        0.1f, -0.1f, 0.2f, -0.2f, 0.0f, 0.3f, -0.3f, 0.1f,
                        -0.1f, 0.2f, -0.2f, 0.0f, 0.3f, -0.3f, 0.1f, -0.1f
                    )
                }
                
                activation = { tensor -> with(tensor) { relu() } }
            }
            
            // Second hidden layer: 16 -> 16 neurons  
            dense(16) {
                // Weights: 16x16 matrix - explicitly defined values
                weights {
                    fromArray(
                        0.5f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f,
                        -0.1f, 0.5f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f,
                        0.2f, -0.1f, 0.5f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f,
                        -0.1f, 0.2f, -0.1f, 0.5f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f,
                        0.2f, -0.1f, 0.2f, -0.1f, 0.5f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f,
                        -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.5f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f,
                        0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.5f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f,
                        -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.5f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f,
                        0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.5f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f,
                        -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.5f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f,
                        0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.5f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f,
                        -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.5f, -0.1f, 0.2f, -0.1f, 0.2f,
                        0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.5f, -0.1f, 0.2f, -0.1f,
                        -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.5f, -0.1f, 0.2f,
                        0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.5f, -0.1f,
                        -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.2f, -0.1f, 0.5f
                    )
                }
                
                // Bias: 16 values - explicitly defined
                bias {
                    fromArray(
                        0.05f, -0.05f, 0.1f, -0.1f, 0.0f, 0.15f, -0.15f, 0.05f,
                        -0.05f, 0.1f, -0.1f, 0.0f, 0.15f, -0.15f, 0.05f, -0.05f
                    )
                }
                
                activation = { tensor -> with(tensor) { relu() } }
            }
            
            // Output layer: 16 -> 1 neuron
            dense(1) {
                // Weights: 1x16 matrix - explicitly defined values
                weights {
                    fromArray(
                        0.3f, -0.2f, 0.4f, -0.1f, 0.5f, -0.3f, 0.2f, -0.4f,
                        0.1f, -0.5f, 0.3f, -0.2f, 0.4f, -0.1f, 0.5f, -0.3f
                    )
                }
                
                // Bias: single value - explicitly defined
                bias {
                    fromArray(0.0f)
                }
                
                // No activation for output layer (linear output)
            }
        }
        
        println("[DEBUG_LOG] Network compiled successfully!")
        
        // Verify the network was built successfully
        assertNotNull(sinNetwork, "Network should not be null")
        assertEquals("MLP", sinNetwork.name, "Network should be an MLP")
        
        println("[DEBUG_LOG] Network name: ${sinNetwork.name}")
        println("[DEBUG_LOG] Number of modules: ${sinNetwork.modules.size}")
        
        // Get the modules and validate they are Linear layers
        val modules = sinNetwork.modules
        
        // Filter out non-Linear modules (Input, Flatten, etc.) and get only Linear modules
        val linearModules = modules.filterIsInstance<Linear<FP32, Float>>()
        assertEquals(3, linearModules.size, "Should have exactly 3 Linear modules")
        
        // Cast modules to ModuleParameters to access weights and biases
        val linearModulesWithParams = linearModules.filterIsInstance<ModuleParameters<FP32, Float>>()
        assertEquals(3, linearModulesWithParams.size, "All Linear modules should implement ModuleParameters")
        
        println("[DEBUG_LOG] Found ${linearModulesWithParams.size} Linear modules with parameters")
        
        // Validate shapes of each layer
        
        // First hidden layer: 1 -> 16 neurons
        val layer1 = linearModulesWithParams[0]
        val weights1 = layer1.params.weights().value
        val bias1 = layer1.params.bias().value
        
        println("[DEBUG_LOG] Layer 1 - Weights shape: ${weights1.shape}, Bias shape: ${bias1.shape}")
        assertEquals(Shape(16, 1), weights1.shape, "First layer weights should be 16x1")
        assertEquals(Shape(16), bias1.shape, "First layer bias should be 16")
        
        // Second hidden layer: 16 -> 16 neurons
        val layer2 = linearModulesWithParams[1]
        val weights2 = layer2.params.weights().value
        val bias2 = layer2.params.bias().value
        
        println("[DEBUG_LOG] Layer 2 - Weights shape: ${weights2.shape}, Bias shape: ${bias2.shape}")
        assertEquals(Shape(16, 16), weights2.shape, "Second layer weights should be 16x16")
        assertEquals(Shape(16), bias2.shape, "Second layer bias should be 16")
        
        // Output layer: 16 -> 1 neuron
        val layer3 = linearModulesWithParams[2]
        val weights3 = layer3.params.weights().value
        val bias3 = layer3.params.bias().value
        
        println("[DEBUG_LOG] Layer 3 - Weights shape: ${weights3.shape}, Bias shape: ${bias3.shape}")
        assertEquals(Shape(1, 16), weights3.shape, "Output layer weights should be 1x16")
        assertEquals(Shape(1), bias3.shape, "Output layer bias should be 1")
        
        println("[DEBUG_LOG] All shape validations passed!")
        
        // Test that the network can perform a forward pass
        val floatData = floatArrayOf(1.0f)
        val byteData = ByteArrayConverter.convertFloatArrayToBytes(floatData)
        val input = fromBytes(FP32, Shape(1, 1), byteData)
        val output = sinNetwork.forward(input)
        
        assertEquals(Shape(1, 1), output.shape, "Output should have shape [1, 1]")
        println("[DEBUG_LOG] Forward pass successful! Output shape: ${output.shape}")
        
        println("[DEBUG_LOG] SinNetwork compilation and shape validation test completed successfully!")
    }
    
    @Test
    fun testNetworkCompilationWithoutErrors() {
        // Simple test to ensure the network compiles without throwing exceptions
        try {
            val network = network<FP32, Float> {
                input(1)
                dense(16) {
                    weights { 
                        fromArray(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f)
                    }
                    bias { 
                        fromArray(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f)
                    }
                }
                dense(16) {
                    weights { 
                        val data = FloatArray(256) { 0.1f }  // 16x16 = 256 values
                        val byteData = ByteArrayConverter.convertFloatArrayToBytes(data)
                        fromBytes(FP32, shape, byteData)
                    }
                    bias { 
                        val data = FloatArray(16) { 0.0f }   // 16 values
                        val byteData = ByteArrayConverter.convertFloatArrayToBytes(data)
                        fromBytes(FP32, shape, byteData)
                    }
                }
                dense(1) {
                    weights { 
                        val data = FloatArray(16) { 0.1f }   // 1x16 = 16 values
                        val byteData = ByteArrayConverter.convertFloatArrayToBytes(data)
                        fromBytes(FP32, shape, byteData)
                    }
                    bias { 
                        fromArray(0.0f)  // 1 value
                    }
                }
            }
            assertNotNull(network, "Network should not be null")
            println("[DEBUG_LOG] Network compilation test passed - no exceptions thrown")
        } catch (e: Exception) {
            fail("Network creation should not throw exceptions: ${e.message}")
        }
    }
}