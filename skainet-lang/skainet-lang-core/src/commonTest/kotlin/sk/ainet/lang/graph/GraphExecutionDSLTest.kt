package sk.ainet.lang.graph

import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.tensor.dsl.Shape1
import sk.ainet.lang.tensor.ops.plus
import sk.ainet.lang.tensor.ops.minus
import sk.ainet.lang.tensor.ops.times
import sk.ainet.lang.tensor.ops.div
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * Test for the complete graph execution DSL implementation.
 * Verifies the exact usage pattern from the issue description works correctly.
 */
class GraphExecutionDSLTest {

    /**
     * Test the exact usage pattern from the issue description:
     * 
     * val a = tensor<FP32,Float> {
     *     Shape1(1) { ones() }
     * }
     * 
     * val b = tensor<FP32,Float> {
     *     Shape1(1) { ones() }
     * }
     * 
     * val graph = exec<FP32,Float> {
     *     a + b
     * }
     */
    @Test
    fun testExactUsagePatternFromIssue() {
        // Create tensors exactly as specified in the issue
        val a = tensor<FP32, Float> {
            shape(Shape1(1)) { ones() }
        }
        
        val b = tensor<FP32, Float> {
            shape(Shape1(1)) { ones() }
        }
        
        // Execute graph operation exactly as specified
        val result: GraphExecutionResult<FP32, Float, Tensor<FP32, Float>> = exec<FP32, Float, Tensor<FP32, Float>> {
            a + b
        }

        
        // Verify the result
        assertNotNull(result.result, "Result should not be null")
        assertEquals(2.0f, result.result.data[0], "Graph should contain one addition operation")
        assertEquals(1, result.graph.nodes.size, "Graph should contain one addition operation")
        
        val addNode = result.graph.nodes.first()
        assertTrue(addNode.id.startsWith("add_"), "Node should be an addition operation")
        assertEquals(2, addNode.inputs.size, "Addition should have 2 inputs")
        assertEquals(1, addNode.outputs.size, "Addition should have 1 output")
        
        // Test passes - graph node created successfully
    }
    
    /**
     * Test multiple operations within exec block
     */
    @Test
    fun testMultipleOperationsInExecBlock() {
        val a = tensor<FP32, Float> {
            shape(Shape1(2)) { ones() }
        }
        
        val b = tensor<FP32, Float> {
            shape(Shape1(2)) { ones() }
        }
        
        val c = tensor<FP32, Float> {
            shape(Shape1(2)) { ones() }
        }
        
        val result = exec<FP32, Float, Tensor<FP32, Float>> {
            val sum = a + b
            val product = sum * c
            product - a  // Final result
        }
        
        // Should have created 3 operations: add, multiply, subtract
        assertEquals(3, result.graph.nodes.size, "Graph should contain three operations")
    }
    
    /**
     * Test that operations outside exec block fail properly
     */
    @Test
    fun testOperationsOutsideExecBlockFail() {
        val a = tensor<FP32, Float> {
            shape(Shape1(1)) { ones() }
        }
        
        val b = tensor<FP32, Float> {
            shape(Shape1(1)) { ones() }
        }
        
        try {
            // This should throw an exception since no graph context is active
            a + b
            fail("Should have thrown an exception")
        } catch (e: IllegalStateException) {
            assertTrue(
                e.message?.contains("No graph execution context available") == true,
                "Should throw context error"
            )
        }
    }
    
    /**
     * Test additional tensor operations
     */
    @Test
    fun testAdditionalTensorOperations() {
        val a = tensor<FP32, Float> {
            shape(Shape1(3)) { ones() }
        }
        
        val b = tensor<FP32, Float> {
            shape(Shape1(3)) { ones() }
        }
        
        val result = exec<FP32, Float, Tensor<FP32, Float>> {
            val sum = a + b
            val diff = a - b
            val product = a * b
            val quotient = a / b
            
            // Return the sum for verification
            sum
        }
        
        // Should have 4 operations recorded
        assertEquals(4, result.graph.nodes.size, "Graph should contain four operations")
        
        // Verify operation types exist in graph
        val nodeIds = result.graph.nodes.map { it.id }
        assertTrue(nodeIds.any { it.startsWith("add_") }, "Should have add operation")
        assertTrue(nodeIds.any { it.startsWith("subtract_") }, "Should have subtract operation")
        assertTrue(nodeIds.any { it.startsWith("multiply_") }, "Should have multiply operation")
        assertTrue(nodeIds.any { it.startsWith("divide_") }, "Should have divide operation")
    }
}