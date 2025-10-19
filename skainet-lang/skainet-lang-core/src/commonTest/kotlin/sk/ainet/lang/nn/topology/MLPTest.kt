package sk.ainet.lang.nn.topology

import sk.ainet.lang.nn.Linear
import sk.ainet.lang.nn.activations.ReLU
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.dsl.with
import sk.ainet.lang.types.FP32
import kotlin.test.*

class MLPTest {

    @Test
    fun testMLPInference() {
        // Create weights and biases for the layers using the tensor DSL
        val factory = DenseTensorDataFactory()
        
        // Layer 1: 1 input -> 16 outputs
        val weights1 = with<FP32, Float>(factory) {
            tensor(16, 1) { shape -> ones() }
        }
        val bias1 = with<FP32, Float>(factory) {
            tensor(1, 16) { shape -> zeros() }
        }
        
        // Layer 2: 16 inputs -> 16 outputs  
        val weights2 = with<FP32, Float>(factory) {
            tensor(16, 16) { shape -> ones() }
        }
        val bias2 = with<FP32, Float>(factory) {
            tensor(1, 16) { shape -> zeros() }
        }
        
        // Layer 3: 16 inputs -> 1 output
        val weights3 = with<FP32, Float>(factory) {
            tensor(1, 16) { shape -> ones() }
        }
        val bias3 = with<FP32, Float>(factory) {
            tensor(1, 1) { shape -> zeros() }
        }
        
        // Create the MLP with 2 hidden layers of 16 neurons each and 1 output
        val mlp = MLP<FP32, Float>(
            Linear(1, 16, "layer1", weights1, bias1),
            ReLU("relu1"),
            Linear(16, 16, "layer2", weights2, bias2), 
            ReLU("relu2"),
            Linear(16, 1, "layer3", weights3, bias3)
        )
        
        // Create test input
        val input = with<FP32, Float>(factory) {
            tensor(1, 1) { shape -> full(2.0f) }
        }
        
        // Perform inference
        val output = mlp.forward(input)
        
        // Basic assertions to verify the test works
        assertNotNull(output)
        assertEquals(Shape(1, 1), output.shape)
    }
}