# MaxPool2d: 2D Max Pooling Layer

## Introduction

The `MaxPool2d` class implements a 2D max pooling layer for neural networks. Max pooling is a downsampling operation that reduces the spatial dimensions of the input data by taking the maximum value in each pooling window. This operation is commonly used in convolutional neural networks (CNNs) to reduce the computational load, extract dominant features, and provide a form of translation invariance.

In the context of neural networks, max pooling layers are typically inserted between successive convolutional layers to progressively reduce the spatial size of the representation, reduce the number of parameters and computation in the network, and control overfitting.

## Class Parameters

The `MaxPool2d` class in the SK-AI-Net library has the following parameters:

```kotlin
class MaxPool2d(
    val kernelSize: Int,
    val stride: Int = kernelSize,
    override val name: String = "MaxPool2d"
) : Module()
```

Let's explore each parameter in detail:

### kernelSize

The size of the pooling window. This is a square window with dimensions `kernelSize × kernelSize`. Common values are 2×2 or 3×3. The pooling operation takes the maximum value from each window and discards the rest of the information.

### stride

The stride parameter controls how the pooling window moves across the input. A stride of 1 means the window moves one pixel at a time, while a stride of 2 means it skips every other pixel. By default, the stride is set to the same value as the kernel size, which is a common practice for pooling layers.

### name

An optional name for the layer, useful for debugging and model visualization.

## Understanding Max Pooling

Max pooling is a form of non-linear downsampling. The operation partitions the input into a set of non-overlapping rectangles and, for each such sub-region, outputs the maximum value.

### Benefits of Max Pooling

1. **Dimensionality Reduction**: Reduces the spatial dimensions of the feature maps, leading to fewer parameters and computations in the network.
2. **Translation Invariance**: Provides a form of translation invariance, making the network more robust to small shifts in the input.
3. **Feature Extraction**: Helps in extracting dominant features by discarding less important information.
4. **Overfitting Control**: Reduces the risk of overfitting by providing a form of regularization.

### Output Size Calculation

The output size can be calculated using the formula:
```
outputSize = (inputSize - kernelSize) / stride + 1
```

### Example

For an input feature map of size 4×4 with a 2×2 kernel and stride of 2:
- The output size will be (4 - 2) / 2 + 1 = 2, resulting in a 2×2 output feature map.
- Each value in the output is the maximum of a 2×2 region in the input.

## Sample Use Cases

### Basic Max Pooling

```kotlin
// Create a simple 1x1x4x4 input tensor (1 batch, 1 channel, 4x4 spatial dimensions)
val input = DoublesTensor(
    Shape(1, 1, 4, 4),
    doubleArrayOf(
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0
    )
)

// Create a MaxPool2d layer with 2x2 kernel and stride=2
val pool = MaxPool2d(
    kernelSize = 2,
    stride = 2
)

// Apply max pooling
val result = pool.forward(input) as DoublesTensor

// Expected output shape: 1x1x2x2 (1 batch, 1 channel, 2x2 spatial dimensions)
println(result.shape)  // Shape(1, 1, 2, 2)

// Expected output values: the maximum value in each 2x2 window
// [6.0, 8.0, 14.0, 16.0]
println(result.elements.toList())
```

### Max Pooling with Different Stride

```kotlin
// Create a simple 1x1x5x5 input tensor
val input = DoublesTensor(
    Shape(1, 1, 5, 5),
    doubleArrayOf(
        1.0, 2.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0,
        16.0, 17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0, 25.0
    )
)

// Create a MaxPool2d layer with 3x3 kernel and stride=1
val pool = MaxPool2d(
    kernelSize = 3,
    stride = 1
)

// Apply max pooling
val result = pool.forward(input) as DoublesTensor

// Expected output shape: 1x1x3x3 (1 batch, 1 channel, 3x3 spatial dimensions)
println(result.shape)  // Shape(1, 1, 3, 3)
```

### Max Pooling with Multiple Channels

```kotlin
// Create a 1x3x4x4 input tensor (1 batch, 3 channels, 4x4 spatial dimensions)
val input = DoublesTensor(
    Shape(1, 3, 4, 4),
    doubleArrayOf(
        // Channel 1
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
        // Channel 2
        17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0,
        25.0, 26.0, 27.0, 28.0,
        29.0, 30.0, 31.0, 32.0,
        // Channel 3
        33.0, 34.0, 35.0, 36.0,
        37.0, 38.0, 39.0, 40.0,
        41.0, 42.0, 43.0, 44.0,
        45.0, 46.0, 47.0, 48.0
    )
)

// Create a MaxPool2d layer with 2x2 kernel and stride=2
val pool = MaxPool2d(
    kernelSize = 2,
    stride = 2
)

// Apply max pooling
val result = pool.forward(input) as DoublesTensor

// Expected output shape: 1x3x2x2 (1 batch, 3 channels, 2x2 spatial dimensions)
println(result.shape)  // Shape(1, 3, 2, 2)
```

### Using MaxPool2d in a Network

```kotlin
// Create a simple CNN for image classification
val network = network {
    // Input layer for 1-channel 28x28 images (e.g., MNIST)
    input(1)
    
    // First convolutional layer
    conv2d {
        outChannels = 16
        kernelSize = 3
        stride = 1
        padding = 1
    }
    
    // Apply ReLU activation
    activation { it.relu() }
    
    // Max pooling layer
    maxPool2d {
        kernelSize = 2
        stride = 2
    }
    
    // Second convolutional layer
    conv2d {
        outChannels = 32
        kernelSize = 3
        stride = 1
        padding = 1
    }
    
    // Apply ReLU activation
    activation { it.relu() }
    
    // Max pooling layer
    maxPool2d {
        kernelSize = 2
        stride = 2
    }
    
    // Flatten the output for the fully connected layer
    flatten()
    
    // Fully connected layer with 10 output units (e.g., for 10 digit classes)
    dense(10)
}

// Use the network for inference
val input = DoublesTensor(Shape(1, 1, 28, 28), /* image data */)
val output = network.forward(input)
```

## Conclusion

The `MaxPool2d` class is an essential component for building convolutional neural networks. By reducing spatial dimensions and extracting dominant features, max pooling helps create more efficient and effective neural network architectures. When designing your network, consider the appropriate kernel size and stride values for your specific application to balance between feature extraction and information preservation.