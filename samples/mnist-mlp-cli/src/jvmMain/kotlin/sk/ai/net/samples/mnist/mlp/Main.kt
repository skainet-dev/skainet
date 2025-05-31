package sk.ai.net.samples.mnist.mlp

import kotlinx.cli.ArgParser
import kotlinx.cli.ArgType
import kotlinx.coroutines.runBlocking
import sk.ai.net.Shape
import sk.ai.net.Tensor
import sk.ai.net.dsl.network
import sk.ai.net.impl.DoublesTensor
import sk.ai.net.io.data.mnist.MNISTConstants
import sk.ai.net.io.data.mnist.MNISTImage
import sk.ai.net.io.data.mnist.MNISTLoaderFactory
import sk.ai.net.nn.Module
import sk.ai.net.nn.activations.ReLU

/**
 * Main entry point for the mnist-mlp-cli application.
 * 
 * This application creates an MLP network and uses it to classify MNIST handwritten digits.
 * It loads the MNIST test data, classifies a subset of the images, and reports the accuracy.
 * 
 * Usage:
 * ```
 * ./gradlew :samples:mnist-mlp-cli:run --args="-m model.gguf -n 100"
 * ```
 * 
 * Options:
 * - `-m, --model`: Path to the model file (required, but not used in this simplified example)
 * - `-n, --num-images`: Number of images to classify (default: 10)
 */
actual fun main(args: Array<String>) {
    val parser = ArgParser("mnist-mlp-cli")
    val numImagesArg by parser.option(
        ArgType.Int,
        shortName = "n",
        fullName = "num-images",
        description = "Number of images to classify (default: 10)"
    )
    val numImages = numImagesArg ?: 10

    parser.parse(args)

    println("MNIST MLP CLI")
    println("=============")
    println("Number of images to classify: $numImages")
    println()

    try {

        // In a real implementation, we would use ModelFormatLoader to load the model
        // But for this simplified example, we'll create an MLP directly
        println("Creating MLP network...")
        val mlp = createMNISTMLP()

        // Load MNIST test data
        println("Loading MNIST test data...")
        val testData = runBlocking {
            val loader = MNISTLoaderFactory.create()
            loader.loadTestData()
        }
        println("Loaded ${testData.size} test images")

        // Classify a subset of the test images
        val numImagesToClassify = minOf(numImages ?: 10, testData.size)
        println("\nClassifying $numImagesToClassify images...")

        var correctPredictions = 0
        for (i in 0 until numImagesToClassify) {
            val image = testData.images[i]
            val prediction = classifyImage(mlp, image)
            val actualLabel = image.label.toInt()
            val isCorrect = prediction == actualLabel

            if (isCorrect) {
                correctPredictions++
            }

            println("Image ${i + 1}: Predicted $prediction, Actual $actualLabel ${if (isCorrect) "✓" else "✗"}")
        }

        // Print accuracy
        val accuracy = correctPredictions.toDouble() / numImagesToClassify * 100
        println("\nAccuracy: $correctPredictions/$numImagesToClassify (${accuracy.toInt()}%)")

    } catch (e: Exception) {
        println("Error: ${e.message}")
        e.printStackTrace()
    }
}

/**
 * Creates an MLP network for MNIST digit classification.
 * 
 * The network architecture is as follows:
 * - Input layer: 784 neurons (28x28 pixels flattened)
 * - First hidden layer: 128 neurons with ReLU activation
 * - Second hidden layer: 128 neurons with ReLU activation
 * - Output layer: 10 neurons (one for each digit 0-9)
 * 
 * This is a simple feedforward neural network that takes a flattened MNIST image as input
 * and outputs a probability distribution over the 10 possible digits. The digit with the
 * highest probability is the predicted digit.
 * 
 * @return The MLP network.
 */
private fun createMNISTMLP(): sk.ai.net.nn.Module {
    // Create the MLP network using the DSL
    return network {
        input(MNISTConstants.IMAGE_PIXELS) // 784 input neurons (28x28 pixels)
        dense(128) { // First hidden layer with 128 neurons
            activation = ReLU()::invoke
        }
        dense(128) { // Second hidden layer with 128 neurons
            activation = ReLU()::invoke
        }
        dense(10) { // Output layer with 10 neurons (one for each digit 0-9)
            // No activation for the output layer (linear)
        }
    }
}

/**
 * Converts a MNIST image to a tensor suitable for input to the MLP network.
 * 
 * The MNIST image is a 28x28 grayscale image stored as a ByteArray.
 * This function:
 * 1. Normalizes the pixel values from [0, 255] to [0, 1]
 * 2. Flattens the 28x28 image into a 784-element vector
 * 3. Creates a tensor with shape [784]
 * 
 * @param image The MNIST image to convert.
 * @return A tensor representing the image.
 */
private fun convertImageToTensor(image: MNISTImage): Tensor {
    // Normalize pixel values to range [0, 1]
    val normalizedPixels = DoubleArray(MNISTConstants.IMAGE_PIXELS) { i ->
        (image.image[i].toInt() and 0xFF) / 255.0
    }

    // Create a tensor with shape [784] (flattened 28x28 image)
    return DoublesTensor(Shape(MNISTConstants.IMAGE_PIXELS), normalizedPixels)
}

/**
 * Classifies a MNIST image using the given MLP network.
 * 
 * This function:
 * 1. Converts the MNIST image to a tensor
 * 2. Passes the tensor through the MLP network
 * 3. Finds the index of the maximum value in the output tensor
 * 4. Returns the index as the predicted digit (0-9)
 * 
 * The output tensor has 10 elements, one for each possible digit.
 * The element with the highest value corresponds to the most likely digit.
 * 
 * @param mlp The MLP network to use for classification.
 * @param image The MNIST image to classify.
 * @return The predicted digit (0-9).
 */
private fun classifyImage(mlp: Module, image: MNISTImage): Int {
    // Convert the image to a tensor
    val inputTensor = convertImageToTensor(image)

    // Forward pass through the network
    val outputTensor = mlp.forward(inputTensor) as DoublesTensor

    // Find the index of the maximum value in the output tensor
    // This is the predicted digit
    val outputElements = outputTensor.elements
    var maxIndex = 0
    var maxValue: Double = outputElements[0]

    for (i in 1 until outputElements.size) {
        if (outputElements[i] > maxValue) {
            maxValue = outputElements[i]
            maxIndex = i
        }
    }

    return maxIndex
}
