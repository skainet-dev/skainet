package sk.ai.net.samples.mnist.mlp

import kotlinx.cli.ArgParser
import kotlinx.cli.ArgType
import kotlinx.cli.required
import sk.ai.net.Shape
import sk.ai.net.Tensor
import sk.ai.net.dsl.network
import sk.ai.net.impl.DoublesTensor
import sk.ai.net.nn.activations.ReLU
import java.io.File

/**
 * Main entry point for the sinus-mlp-cli application.
 * 
 * This application loads weights from a GGUF or SafeTensors file and uses an MLP network
 * to approximate the sine function.
 */
actual fun main(args: Array<String>) {
    val parser = ArgParser("sinus-mlp-cli")
    val modelPath by parser.option(
        ArgType.String,
        shortName = "m",
        fullName = "model",
        description = "Path to the model file (GGUF or SafeTensors)"
    ).required()

    parser.parse(args)

    println("MNIST MLP CLI")
    println("=============")
    println("Model file: $modelPath")
    println()

    try {
        // Check if the model file exists
        val modelFile = File(modelPath)
        if (!modelFile.exists()) {
            println("Error: Model file not found: $modelPath")
            return
        }

        // In a real implementation, we would use ModelFormatLoader to load the model
        // But for this simplified example, we'll create a simple MLP directly
        println("Creating MLP network...")
        val mlp = createSimpleMLP()


    } catch (e: Exception) {
        println("Error: ${e.message}")
        e.printStackTrace()
    }
}

/**
 * Creates a simple MLP network for sine approximation.
 * 
 * The network has one input (angle), two hidden layers with 10 neurons each, and one output neuron.
 * 
 * @return The MLP network.
 */
private fun createSimpleMLP(): sk.ai.net.nn.Module {
    // Create a context for the DSL
    // Create the MLP network using the DSL
    return network {
        input(1) // One input neuron for the angle
        dense(10) { // First hidden layer with 10 neurons
            activation = ReLU()::invoke
        }
        dense(10) { // Second hidden layer with 10 neurons
            activation = ReLU()::invoke
        }
        dense(1) { // Output layer with 1 neuron
            // No activation for the output layer (linear)
        }
    }
}

/**
 * Creates an input tensor for the given angle.
 * 
 * @param angle The angle in radians.
 * @return A tensor representing the angle.
 */
private fun createInputTensor(angle: Double): Tensor {
    return DoublesTensor (Shape(1), doubleArrayOf(angle))
}
