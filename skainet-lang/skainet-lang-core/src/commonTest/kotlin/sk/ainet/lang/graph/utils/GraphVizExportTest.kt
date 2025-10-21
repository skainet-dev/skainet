package sk.ainet.lang.graph.utils

import sk.ainet.lang.graph.*
import kotlin.test.Test
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.test.assertEquals

/**
 * Test for GraphViz export functionality
 */
class GraphVizExportTest {

    private class TestOperation(
        override val name: String,
        override val type: String,
        override val parameters: Map<String, Any> = emptyMap()
    ) : BaseOperation(name, type, parameters) {
        
        override fun <T : sk.ainet.lang.types.DType, V> execute(inputs: List<sk.ainet.lang.tensor.Tensor<T, V>>): List<sk.ainet.lang.tensor.Tensor<T, V>> {
            return inputs // Pass through for testing
        }
        
        override fun validateInputs(inputs: List<TensorSpec>): ValidationResult {
            return ValidationResult.Valid
        }
        
        override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
            return inputs // Pass through for testing
        }
        
        override fun clone(newParameters: Map<String, Any>): Operation {
            return TestOperation(name, type, newParameters)
        }
    }

    @Test
    fun testBasicGraphVizExport() {
        println("[DEBUG_LOG] Testing basic GraphViz export functionality")
        
        // Create a simple compute graph
        val graph = DefaultComputeGraph()
        
        // Create test operations
        val inputOp = TestOperation("input", "input")
        val processOp = TestOperation("process", "compute", mapOf("kernel_size" to 3, "stride" to 1))
        val outputOp = TestOperation("output", "output")
        
        // Create test nodes
        val inputNode = GraphNode(
            id = "input_node",
            operation = inputOp,
            inputs = emptyList(),
            outputs = listOf(TensorSpec("input_out", listOf(1, 10), "FP32"))
        )
        
        val processNode = GraphNode(
            id = "process_node", 
            operation = processOp,
            inputs = listOf(TensorSpec("process_in", listOf(1, 10), "FP32")),
            outputs = listOf(TensorSpec("process_out", listOf(1, 5), "FP32"))
        )
        
        val outputNode = GraphNode(
            id = "output_node",
            operation = outputOp,
            inputs = listOf(TensorSpec("output_in", listOf(1, 5), "FP32")),
            outputs = listOf(TensorSpec("output_out", listOf(1, 5), "FP32"))
        )
        
        // Add nodes to graph
        graph.addNode(inputNode)
        graph.addNode(processNode)
        graph.addNode(outputNode)
        
        // Add edges to connect them
        graph.addEdge(GraphEdge("edge1", inputNode, processNode, 0, 0, inputNode.outputs.first()))
        graph.addEdge(GraphEdge("edge2", processNode, outputNode, 0, 0, processNode.outputs.first()))
        
        println("[DEBUG_LOG] Created graph with ${graph.nodes.size} nodes and ${graph.edges.size} edges")
        
        // Test full graph export
        val dotGraph = drawDot(graph)
        assertNotNull(dotGraph, "DOT graph should not be null")
        assertNotNull(dotGraph.content, "DOT content should not be null")
        assertTrue(dotGraph.content.isNotEmpty(), "DOT content should not be empty")
        
        println("[DEBUG_LOG] Full graph DOT export:")
        println(dotGraph.content)
        
        // Verify DOT content structure
        assertTrue(dotGraph.content.contains("digraph {"), "Should contain digraph declaration")
        assertTrue(dotGraph.content.contains("rankdir=LR"), "Should contain rankdir setting")
        assertTrue(dotGraph.content.contains("input_node"), "Should contain input node")
        assertTrue(dotGraph.content.contains("process_node"), "Should contain process node") 
        assertTrue(dotGraph.content.contains("output_node"), "Should contain output node")
        assertTrue(dotGraph.content.contains("->"), "Should contain edges")
        assertTrue(dotGraph.content.contains("}"), "Should contain closing brace")
        
        // Test parameters are included
        assertTrue(dotGraph.content.contains("kernel_size"), "Should include operation parameters")
        
        println("[DEBUG_LOG] Basic GraphViz export test passed")
    }
    
    @Test
    fun testSubsetGraphVizExport() {
        println("[DEBUG_LOG] Testing subset GraphViz export functionality")
        
        // Create a compute graph
        val graph = DefaultComputeGraph()
        
        // Create test nodes
        val node1 = GraphNode("node1", TestOperation("op1", "type1"), emptyList(), listOf(TensorSpec("out1", listOf(1), "FP32")))
        val node2 = GraphNode("node2", TestOperation("op2", "type2"), listOf(TensorSpec("in2", listOf(1), "FP32")), listOf(TensorSpec("out2", listOf(1), "FP32")))
        val node3 = GraphNode("node3", TestOperation("op3", "type3"), listOf(TensorSpec("in3", listOf(1), "FP32")), listOf(TensorSpec("out3", listOf(1), "FP32")))
        
        graph.addNode(node1)
        graph.addNode(node2)
        graph.addNode(node3)
        
        graph.addEdge(GraphEdge("edge1", node1, node2, 0, 0, node1.outputs.first()))
        graph.addEdge(GraphEdge("edge2", node2, node3, 0, 0, node2.outputs.first()))
        
        // Test subset export (only from node3 backward)
        val dotGraphSubset = drawDot(graph, listOf(node3))
        assertNotNull(dotGraphSubset, "Subset DOT graph should not be null")
        assertTrue(dotGraphSubset.content.isNotEmpty(), "Subset DOT content should not be empty")
        
        println("[DEBUG_LOG] Subset DOT export:")
        println(dotGraphSubset.content)
        
        // All nodes should be included since node3 depends on all previous nodes
        assertTrue(dotGraphSubset.content.contains("node1"), "Should contain node1 in subset")
        assertTrue(dotGraphSubset.content.contains("node2"), "Should contain node2 in subset")
        assertTrue(dotGraphSubset.content.contains("node3"), "Should contain node3 in subset")
        
        println("[DEBUG_LOG] Subset GraphViz export test passed")
    }
    
    @Test
    fun testDifferentRankDirections() {
        println("[DEBUG_LOG] Testing different rank directions")
        
        val graph = DefaultComputeGraph()
        val node = GraphNode("test", TestOperation("test", "test"), emptyList(), listOf(TensorSpec("out", listOf(1), "FP32")))
        graph.addNode(node)
        
        // Test LR direction (default)
        val dotLR = drawDot(graph, "LR")
        assertTrue(dotLR.content.contains("rankdir=LR"), "Should set LR rank direction")
        
        // Test TB direction  
        val dotTB = drawDot(graph, "TB")
        assertTrue(dotTB.content.contains("rankdir=TB"), "Should set TB rank direction")
        
        println("[DEBUG_LOG] Rank direction test passed")
    }
}