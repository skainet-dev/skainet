# Conv2d: 2D Convolution Layer

## Introduction

The `Conv2d` class implements a 2D convolution layer for neural networks. Convolution is a fundamental operation in deep learning, especially for processing grid-like data such as images. The 2D convolution operation slides a kernel (or filter) over the input data, computing the element-wise product and sum at each position.

In the context of neural networks, convolution layers are used to automatically and adaptively learn spatial hierarchies of features from input data. For image processing, early convolution layers might detect simple features like edges, while deeper layers can recognize more complex patterns like textures or even entire objects.

### Why Conv2D is Differentiable
A Conv2D layer is differentiable because its operations are mathematically well-defined and have computable derivatives:

  * Convolution: For an input tensor $ X $ (e.g., shape (1, 28, 28) for MNIST), a kernel $ W $ (e.g., 5x5), and bias $ b $, the output at position $(i, j)$ for output channel $ k $ is:
$$Y_k[i, j] = \sum_{m,n,c} X[c, i+m, j+n] \cdot W_k[c, m, n] + b_k$$
where $ m, n $ iterate over the kernel size, and $ c $ iterates over input channels.

This is a linear operation (sum of products plus a constant), which is differentiable.
Partial derivatives with respect to weights ($ \frac{\partial Y}{\partial W} $) and input ($ \frac{\partial Y}{\partial X} $) are straightforward:

$ \frac{\partial Y_k[i, j]}{\partial W_k[c, m, n]} = X[c, i+m, j+n] $
$ \frac{\partial Y_k[i, j]}{\partial X[c, i+m, j+n]} = W_k[c, m, n] $
$ \frac{\partial Y_k[i, j]}{\partial b_k} = 1 $




Stride and Padding: Stride controls the step size of the kernel, and padding adds zeros around the input. Both are fixed operations that don’t affect differentiability, as they only modify indexing.
Output: The output $ Y $ is passed to the next layer (e.g., ReLU in your CNN), which is also differentiable.

3. Role in Learning (Backpropagation)
   The Conv2D layer supports learning through backpropagation, where gradients of the loss function $ L $ with respect to the weights and biases are computed and used to update parameters:

Forward Pass:

Input (e.g., MNIST image) passes through the Conv2D layer, producing feature maps (e.g., 16 channels of 28x28 for conv1, due to padding=2).
These feature maps are transformed by subsequent layers (ReLU, MaxPool2D, etc.) and eventually produce a prediction (e.g., probabilities for 10 digit classes).
The loss $ L $ (e.g., cross-entropy) compares the prediction to the true label.


Backward Pass:

The gradient of the loss with respect to the output of Conv2D ($ \frac{\partial L}{\partial Y} $) is received from the next layer (e.g., ReLU).
Gradients for weights and biases are computed:

Weight gradient: $ \frac{\partial L}{\partial W_k[c, m, n]} = \sum_{i,j} \frac{\partial L}{\partial Y_k[i, j]} \cdot X[c, i+m, j+n] $
Bias gradient: $ \frac{\partial L}{\partial b_k} = \sum_{i,j} \frac{\partial L}{\partial Y_k[i, j]} $
Input gradient (for backprop to earlier layers): $ \frac{\partial L}{\partial X[c, i+m, j+n]} = \sum_k \frac{\partial L}{\partial Y_k[i, j]} \cdot W_k[c, m, n] $


These gradients are computed efficiently using cross-correlation (a convolution-like operation).


Parameter Update:

An optimizer (e.g., SGD, Adam) updates the weights and biases:
$$W \gets W - \eta \cdot \frac{\partial L}{\partial W}, \quad b \gets b - \eta \cdot \frac{\partial L}{\partial b}$$
where $ \eta $ is the learning rate.
This adjusts the kernels to better detect features (e.g., edges, shapes in MNIST digits) that minimize the loss.

## Class Parameters

The `Conv2d` class in the SK-AI-Net library has the following parameters:

```kotlin
class Conv2d(
    val inChannels: Int,
    val outChannels: Int,
    val kernelSize: Int,
    val stride: Int = 1,
    val padding: Int = 0,
    useBias: Boolean = true,
    name: String = "Conv2d"
) : Module()
```

Let's explore each parameter in detail:

### inChannels

The number of input channels (or feature maps). For RGB images, this would typically be 3 (red, green, blue). For grayscale images, this would be 1. For layers deeper in the network, this corresponds to the number of feature maps produced by the previous layer.

### outChannels

The number of output channels (or feature maps) that the convolution layer will produce. This determines how many different filters will be applied to the input. Each filter learns to detect different features in the input.

### kernelSize

The size of the convolution kernel (or filter). This is a square kernel with dimensions `kernelSize × kernelSize`. Common values are 3×3, 5×5, or 7×7. Larger kernels can capture more spatial context but require more computation and may lead to overfitting.

### stride

The stride parameter controls how the kernel moves across the input. A stride of 1 means the kernel moves one pixel at a time, while a stride of 2 means it skips every other pixel. Larger strides result in smaller output dimensions and can be used for downsampling.

### padding

Padding adds extra pixels around the input data before applying the convolution. This is useful for preserving the spatial dimensions of the input. Without padding, the output dimensions would be smaller than the input. Common padding strategies include:

- `padding = 0` (no padding): Output size will be smaller than input
- `padding = kernelSize / 2` (same padding): Output size will be the same as input (for stride=1)

### useBias

A boolean flag indicating whether to include a bias term in the convolution. When `true`, a learnable bias is added to each output channel after the convolution operation.

### name

An optional name for the layer, useful for debugging and model visualization.

## Understanding Padding

Padding is crucial for controlling the spatial dimensions of the output. Without padding, each convolution operation reduces the size of the feature map, which can be problematic for deep networks.

### Effects of Padding

- **No padding (padding = 0)**: The output dimensions will be smaller than the input. Specifically, if the input has spatial dimensions H×W, the output will have dimensions (H - kernelSize + 1)×(W - kernelSize + 1) (for stride=1).
- **Same padding (padding = kernelSize / 2)**: The output dimensions will be the same as the input (for stride=1). This is often desirable to maintain spatial information.
- **Valid padding (padding < kernelSize / 2)**: The output dimensions will be smaller than the input, but some padding is still applied.

### Example

For an input image of size 28×28 with a 3×3 kernel:

- With no padding (padding = 0), the output size will be 26×26
- With padding = 1, the output size will be 28×28 (same as input)

## Understanding Strides

The stride parameter determines how the kernel moves across the input data. It affects both the computational efficiency and the output dimensions.

### Effects of Stride

- **stride = 1**: The kernel moves one pixel at a time, resulting in the maximum possible output size.
- **stride > 1**: The kernel skips pixels, resulting in a smaller output size. This can be used for downsampling or reducing computational requirements.

### Output Size Calculation

The output size can be calculated using the formula:
```
outputSize = (inputSize + 2 * padding - kernelSize) / stride + 1
```

### Example

For an input image of size 28×28 with a 3×3 kernel:

- With padding = 1 and stride = 1, the output size will be 28×28
- With padding = 1 and stride = 2, the output size will be 14×14 (downsampled by a factor of 2)

## Understanding Filters/Kernels

Filters (or kernels) are the learnable parameters in a convolution layer. Each filter is responsible for detecting specific features in the input data.

### Role of Filters

- Each filter is a small matrix (e.g., 3×3) that is convolved with the input
- The values in the filter are learned during training
- Different filters learn to detect different features (e.g., edges, textures, patterns)
- The number of filters determines the number of output channels

### Filter Dimensions

For a `Conv2d` layer with `inChannels` input channels and `outChannels` output channels, the total number of parameters in the filters is:
```
parameters = outChannels * inChannels * kernelSize * kernelSize
```

If `useBias` is true, there are an additional `outChannels` bias parameters.

## Sample Use Cases

### Basic Convolution

```kotlin
// Create a simple 1x1x3x3 input tensor (1 batch, 1 channel, 3x3 spatial dimensions)
val input = DoublesTensor(
    Shape(1, 1, 3, 3),
    doubleArrayOf(
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    )
)

// Create a Conv2d layer with 1 input channel, 1 output channel, 2x2 kernel, no stride, no padding
val conv = Conv2d(
    inChannels = 1,
    outChannels = 1,
    kernelSize = 2,
    stride = 1,
    padding = 0,
    useBias = true
)

// Apply convolution
val result = conv.forward(input) as DoublesTensor

// Expected output shape: 1x1x2x2 (1 batch, 1 channel, 2x2 spatial dimensions)
println(result.shape)  // Shape(1, 1, 2, 2)
```

### Convolution with Padding

```kotlin
// Create a simple 1x1x3x3 input tensor
val input = DoublesTensor(
    Shape(1, 1, 3, 3),
    doubleArrayOf(
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    )
)

// Create a Conv2d layer with padding=1
val conv = Conv2d(
    inChannels = 1,
    outChannels = 1,
    kernelSize = 3,
    stride = 1,
    padding = 1,
    useBias = true
)

// Apply convolution
val result = conv.forward(input) as DoublesTensor

// Expected output shape: 1x1x3x3 (padding preserves spatial dimensions)
println(result.shape)  // Shape(1, 1, 3, 3)
```

### Convolution with Stride

```kotlin
// Create a simple 1x1x4x4 input tensor
val input = DoublesTensor(
    Shape(1, 1, 4, 4),
    doubleArrayOf(
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0
    )
)

// Create a Conv2d layer with stride=2
val conv = Conv2d(
    inChannels = 1,
    outChannels = 1,
    kernelSize = 2,
    stride = 2,
    padding = 0,
    useBias = true
)

// Apply convolution
val result = conv.forward(input) as DoublesTensor

// Expected output shape: 1x1x2x2 (stride=2 reduces spatial dimensions by half)
println(result.shape)  // Shape(1, 1, 2, 2)
```

### Multiple Input and Output Channels

```kotlin
// Create a 1x2x3x3 input tensor (1 batch, 2 channels, 3x3 spatial dimensions)
val input = DoublesTensor(
    Shape(1, 2, 3, 3),
    doubleArrayOf(
        // Channel 1
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        // Channel 2
        9.0, 8.0, 7.0,
        6.0, 5.0, 4.0,
        3.0, 2.0, 1.0
    )
)

// Create a Conv2d layer with 2 input channels, 3 output channels
val conv = Conv2d(
    inChannels = 2,
    outChannels = 3,
    kernelSize = 2,
    stride = 1,
    padding = 0,
    useBias = true
)

// Apply convolution
val result = conv.forward(input) as DoublesTensor

// Expected output shape: 1x3x2x2 (1 batch, 3 output channels, 2x2 spatial dimensions)
println(result.shape)  // Shape(1, 3, 2, 2)
```

### Using Conv2d in a Network

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

The `Conv2d` class is a powerful tool for building convolutional neural networks. By understanding the parameters and their effects, you can design effective architectures for various computer vision tasks. Experiment with different kernel sizes, strides, and padding values to find the configuration that works best for your specific application.