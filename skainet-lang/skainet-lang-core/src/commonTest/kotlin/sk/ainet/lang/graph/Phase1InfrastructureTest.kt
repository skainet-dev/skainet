package sk.ainet.lang.graph

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.test.assertFalse

/**
 * Tests for Phase 1: Core Infrastructure implementation
 */
class Phase1InfrastructureTest {
    
    @Test
    fun testComputeGraphBasicOperations() {
        val graph = DefaultComputeGraph()
        
        // Create test operations
        val addOp = TestOperation("add", "math")
        val mulOp = TestOperation("mul", "math")
        
        // Create test nodes
        val node1 = GraphNode(
            id = "node1",
            operation = addOp,
            inputs = listOf(
                TensorSpec("input1", listOf(2, 3), "FP32"),
                TensorSpec("input2", listOf(2, 3), "FP32")
            ),
            outputs = listOf(TensorSpec("output1", listOf(2, 3), "FP32"))
        )
        
        val node2 = GraphNode(
            id = "node2", 
            operation = mulOp,
            inputs = listOf(TensorSpec("input3", listOf(2, 3), "FP32")),
            outputs = listOf(TensorSpec("output2", listOf(2, 3), "FP32"))
        )
        
        // Test adding nodes
        graph.addNode(node1)
        graph.addNode(node2)
        
        assertEquals(2, graph.nodes.size)
        assertTrue(graph.nodes.contains(node1))
        assertTrue(graph.nodes.contains(node2))
        
        // Test adding edge
        val edge = GraphEdge(
            id = "edge1",
            source = node1,
            destination = node2,
            tensorSpec = node1.outputs.first()
        )
        
        graph.addEdge(edge)
        assertEquals(1, graph.edges.size)
        
        // Test graph validation
        val validation = graph.validate()
        assertTrue(validation is ValidationResult.Valid)
    }
    
    @Test
    fun testOperationRegistry() {
        val registry = DefaultOperationRegistry()
        val factory = TestOperationFactory()
        
        // Test registration
        registry.registerOperation("test_add", factory)
        assertTrue(registry.isRegistered("test_add"))
        assertEquals(1, registry.registeredOperations.size)
        
        // Test operation creation
        val operation = registry.createOperation("test_add", mapOf("param1" to "value1"))
        assertNotNull(operation)
        assertEquals("test_add", operation.name)
        assertEquals("math", operation.type)
        
        // Test metadata
        val metadata = registry.getOperationMetadata("test_add")
        assertNotNull(metadata)
        assertEquals("test_add", metadata.name)
        assertEquals("math", metadata.type)
    }
    
    @Test
    fun testExecutionTape() {
        val tape = DefaultExecutionTape()
        
        // Test initial state
        assertFalse(tape.isRecording)
        assertEquals(0, tape.operations.size)
        
        // Test recording
        tape.startRecording()
        assertTrue(tape.isRecording)
        
        // TODO: Add actual tensor creation once tensor infrastructure is available
        // For now, test with mock operations
        
        tape.stopRecording()
        assertFalse(tape.isRecording)
    }
    
    @Test
    fun testTapeStack() {
        val stack = DefaultTapeStack()
        val tape1 = DefaultExecutionTape()
        val tape2 = DefaultExecutionTape()
        
        // Test initial state
        assertEquals(null, stack.currentTape)
        assertEquals(0, stack.tapes.size)
        assertFalse(stack.isRecording())
        
        // Test pushing tapes
        stack.pushTape(tape1)
        assertEquals(tape1, stack.currentTape)
        assertEquals(1, stack.tapes.size)
        
        stack.pushTape(tape2)
        assertEquals(tape2, stack.currentTape)
        assertEquals(2, stack.tapes.size)
        
        // Test popping tapes
        val poppedTape = stack.popTape()
        assertEquals(tape2, poppedTape)
        assertEquals(tape1, stack.currentTape)
        assertEquals(1, stack.tapes.size)
    }
    
    @Test
    fun testExecutionContext() {
        val context = DefaultExecutionContext()
        
        // Test initial state
        assertEquals(ExecutionMode.EAGER, context.executionMode)
        assertEquals(DeviceType.CPU, context.device.type)
        assertFalse(context.isRecording)
        
        // Test mode switching
        context.switchToGraph()
        assertEquals(ExecutionMode.GRAPH, context.executionMode)
        
        context.switchToEager()
        assertEquals(ExecutionMode.EAGER, context.executionMode)
        
        // Test device switching
        val gpuDevice = Device(DeviceType.GPU, 0)
        context.setDevice(gpuDevice)
        assertEquals(DeviceType.GPU, context.device.type)
        assertEquals(0, context.device.id)
        
        // Test memory info
        val memoryInfo = context.getMemoryInfo()
        assertTrue(memoryInfo.totalMemory > 0)
        assertTrue(memoryInfo.usedMemory >= 0)
        assertTrue(memoryInfo.freeMemory >= 0)
    }
    
    @Test
    fun testTopologicalSorting() {
        val graph = DefaultComputeGraph()
        
        // Create a simple chain: A -> B -> C
        val nodeA = GraphNode("A", TestOperation("opA", "test"), emptyList(), listOf(TensorSpec("outA", listOf(1), "FP32")))
        val nodeB = GraphNode("B", TestOperation("opB", "test"), listOf(TensorSpec("inB", listOf(1), "FP32")), listOf(TensorSpec("outB", listOf(1), "FP32")))
        val nodeC = GraphNode("C", TestOperation("opC", "test"), listOf(TensorSpec("inC", listOf(1), "FP32")), emptyList())
        
        graph.addNode(nodeA)
        graph.addNode(nodeB)
        graph.addNode(nodeC)
        
        val edgeAB = GraphEdge("AB", nodeA, nodeB, tensorSpec = nodeA.outputs.first())
        val edgeBC = GraphEdge("BC", nodeB, nodeC, tensorSpec = nodeB.outputs.first())
        
        graph.addEdge(edgeAB)
        graph.addEdge(edgeBC)
        
        // Test topological order
        val topOrder = graph.getTopologicalOrder()
        assertEquals(3, topOrder.size)
        assertEquals("A", topOrder[0].id)
        assertEquals("B", topOrder[1].id)
        assertEquals("C", topOrder[2].id)
    }
}

/**
 * Test implementation of Operation for testing purposes
 */
class TestOperation(
    override val name: String,
    override val type: String,
    override val parameters: Map<String, Any> = emptyMap()
) : BaseOperation(name, type, parameters) {
    
    override fun <T : DType, V> execute(inputs: List<Tensor<T, V>>): List<Tensor<T, V>> {
        // Mock execution - just return inputs for now
        return inputs
    }
    
    override fun validateInputs(inputs: List<TensorSpec>): ValidationResult {
        return ValidationResult.Valid
    }
    
    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        return inputs // Mock - just return inputs as outputs
    }
    
    override fun clone(newParameters: Map<String, Any>): Operation {
        return TestOperation(name, type, newParameters)
    }
}

/**
 * Test implementation of OperationFactory
 */
class TestOperationFactory : OperationFactory {
    
    override fun create(parameters: Map<String, Any>): Operation {
        return TestOperation("test_add", "math", parameters)
    }
    
    override fun getMetadata(): OperationMetadata {
        return OperationMetadata(
            name = "test_add",
            type = "math",
            description = "Test addition operation",
            inputSpecs = listOf(
                ParameterSpec("input1", "Tensor<FP32>"),
                ParameterSpec("input2", "Tensor<FP32>")
            ),
            outputSpecs = listOf(
                ParameterSpec("output", "Tensor<FP32>")
            ),
            parameterSpecs = emptyList()
        )
    }
}