package sk.ainet.nn.reflection

import sk.ainet.core.tensor.DType
import sk.ainet.core.tensor.FP32
import sk.ainet.core.tensor.Shape
import sk.ainet.core.tensor.Tensor
import sk.ainet.core.tensor.backend.CpuTensorFP32
import sk.ainet.nn.Module
import sk.ainet.nn.dsl.network
import sk.ainet.nn.topology.ModuleParameter
import sk.ainet.nn.topology.ModuleParameters
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ModelSummaryTest {

    @Test
    fun testBasicSummaryFunctionality() {
        // Create a simple test module without parameters using DSL
        val emptyModule = network<FP32, Float> {
            // Empty network - no layers added
        }

        // Test that Summary can be created and used
        val summary = Summary<FP32, Float>()
        val inputShape = Shape(intArrayOf(1, 10))
        val nodes = summary.summary(emptyModule, inputShape)

        // Empty module should produce empty summary
        assertEquals(0, nodes.size)
    }

    @Test
    fun testPrintSummaryFormat() {
        // Create a simple NodeSummary for testing
        val testNodes = listOf(
            NodeSummary("layer1", Shape(intArrayOf(10, 5)), Shape(intArrayOf(10, 3)), 18L),
            NodeSummary("layer2", Shape(intArrayOf(10, 3)), Shape(intArrayOf(10, 1)), 4L)
        )

        val summary = Summary<FP32, Float>()
        val summaryString = summary.printSummary(testNodes)

        // Check that the summary contains expected elements
        assertTrue(summaryString.contains("layer1"))
        assertTrue(summaryString.contains("layer2"))
        assertTrue(summaryString.contains("18"))
        assertTrue(summaryString.contains("4"))
        assertTrue(summaryString.contains("Layer (type)"))
        assertTrue(summaryString.contains("Output Shape"))
        assertTrue(summaryString.contains("Param #"))
    }

    @Test
    fun testDescribeExtensionFunction() {
        // Test the describe extension function with empty module using DSL
        val testModule = network<FP32, Float> {
            // Empty network - no layers added
        }

        val inputShape = Shape(intArrayOf(2, 5))
        val description = testModule.describe(inputShape)

        // Should return formatted table string
        assertTrue(description.contains("Layer (type)"))
        assertTrue(description.contains("Output Shape"))
        assertTrue(description.contains("Param #"))
    }

    @Test
    fun testNodeSummaryCreation() {
        // Test NodeSummary data class
        val inputShape = Shape(intArrayOf(32, 784))
        val outputShape = Shape(intArrayOf(32, 10))
        val node = NodeSummary("TestLayer", inputShape, outputShape, 7850L)

        assertEquals("TestLayer", node.name)
        assertEquals(inputShape, node.input)
        assertEquals(outputShape, node.output)
        assertEquals(7850L, node.params)
    }

    @Test
    fun testSummaryWithNestedEmptyModules() {
        // Test with nested empty modules using DSL
        val innerModule = network<FP32, Float> {
            // Empty inner network - no layers added
        }

        val outerModule = network<FP32, Float> {
            // Empty outer network - no layers added
            sequential {
                // Could add inner module here, but for testing empty modules we keep it simple
            }
        }

        val summary = Summary<FP32, Float>()
        val inputShape = Shape(intArrayOf(1, 10))
        val nodes = summary.summary(outerModule, inputShape)

        // Should find no parameterized modules
        assertEquals(0, nodes.size)
    }

    @Test
    fun testMLP() {
        val net = network<FP32, Float> {
            input(1)  // Single input for x value

            // First hidden layer: 1 -> 16 neurons
            dense(16) {
                // Weights: 16x1 matrix - explicitly defined values
                weights { shape ->
                    CpuTensorFP32.fromArray(
                        shape,
                        floatArrayOf(
                            0.5f, -0.3f, 0.8f, -0.2f, 0.6f, -0.4f, 0.7f, -0.1f,
                            0.9f, -0.5f, 0.3f, -0.7f, 0.4f, -0.6f, 0.2f, -0.8f
                        )
                    )
                }

                // Bias: 16 values - explicitly defined
                bias { shape ->
                    CpuTensorFP32.fromArray(
                        shape,
                        floatArrayOf(
                            0.1f, -0.1f, 0.2f, -0.2f, 0.0f, 0.3f, -0.3f, 0.1f,
                            -0.1f, 0.2f, -0.2f, 0.0f, 0.3f, -0.3f, 0.1f, -0.1f
                        )
                    )
                }

                activation = { tensor ->
                    with(tensor) { relu() }
                }
            }

            // Second hidden layer: 16 -> 16 neurons
            dense(16) {
                // Weights: 16x16 matrix - explicitly defined values
                weights { shape ->
                    CpuTensorFP32.fromArray(
                        shape,
                        floatArrayOf(
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
                    )
                }

                // Bias: 16 values - explicitly defined
                bias { shape ->
                    CpuTensorFP32.fromArray(
                        shape,
                        floatArrayOf(
                            0.05f, -0.05f, 0.1f, -0.1f, 0.0f, 0.15f, -0.15f, 0.05f,
                            -0.05f, 0.1f, -0.1f, 0.0f, 0.15f, -0.15f, 0.05f, -0.05f
                        )
                    )
                }
                activation = { tensor ->
                    with(tensor) { relu() }
                }
            }

            // Output layer: 16 -> 1 neuron
            dense(1) {
                // Weights: 1x16 matrix - explicitly defined values
                weights { shape ->
                    CpuTensorFP32.fromArray(
                        shape,
                        floatArrayOf(
                            0.3f, -0.2f, 0.4f, -0.1f, 0.5f, -0.3f, 0.2f, -0.4f,
                            0.1f, -0.5f, 0.3f, -0.2f, 0.4f, -0.1f, 0.5f, -0.3f
                        )
                    )
                }

                // Bias: single value - explicitly defined
                bias { shape ->
                    CpuTensorFP32.fromArray(shape, floatArrayOf(0.0f))
                }

                // No activation for output layer (linear output)
            }
        }
        val summary = Summary<FP32, Float>()
        val inputShape = Shape(intArrayOf(1, 10))
        val nodes = summary.summary(net, inputShape)
        val str = net.describe(inputShape)
        assertEquals(str, "")


        // Should find no parameterized modules
        assertEquals(3, nodes.size)



    }
}