package sk.ainet.nn.topology

import sk.ainet.core.tensor.Shape
import sk.ainet.core.tensor.backend.CpuTensorFP32
import sk.ainet.nn.Linear
import sk.ainet.nn.activations.ReLU
import kotlin.test.*

class MLPTest {

    @Test
    fun testMLPInference() {
        // Create weights and biases for the layers
        // Layer 1: 1 input -> 16 outputs
        val weights1 = CpuTensorFP32.ones(Shape(16, 1))  // 16x1 weight matrix
        val bias1 = CpuTensorFP32.zeros(Shape(1, 16))    // 1x16 bias values
        
        // Layer 2: 16 inputs -> 16 outputs  
        val weights2 = CpuTensorFP32.ones(Shape(16, 16)) // 16x16 weight matrix
        val bias2 = CpuTensorFP32.zeros(Shape(1, 16))    // 1x16 bias values
        
        // Layer 3: 16 inputs -> 1 output
        val weights3 = CpuTensorFP32.ones(Shape(1, 16))  // 1x16 weight matrix
        val bias3 = CpuTensorFP32.zeros(Shape(1, 1))     // 1x1 bias value
        
        // Create the MLP with 2 hidden layers of 16 neurons each and 1 output
        val mlp = MLP(
            Linear(1, 16, "layer1", weights1, bias1),
            ReLU("relu1"),
            Linear(16, 16, "layer2", weights2, bias2), 
            ReLU("relu2"),
            Linear(16, 1, "layer3", weights3, bias3)
        )
        
        // Create input tensor with 1 float value (batch_size=1, features=1)
        val input = CpuTensorFP32.fromArray(Shape(1, 1), floatArrayOf(2.0f))
        
        // Perform inference
        val output = with(mlp) { 
            input.forward(input)
        }
        
        // Verify the output shape and that inference ran without errors
        assertEquals(Shape(1, 1), output.shape)
        assertTrue(output[0, 0] > 0.0f, "Output should be positive due to ReLU activations and positive weights")
        
        // Test with different input
        val input2 = CpuTensorFP32.fromArray(Shape(1, 1), floatArrayOf(-1.0f))
        val output2 = with(mlp) {
            input2.forward(input2)
        }
        
        assertEquals(Shape(1, 1), output2.shape)
        // With negative input and ReLU, intermediate values may be zero, but final output depends on the architecture
    }
}