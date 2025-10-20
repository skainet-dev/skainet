package sk.ainet.lang.tensor.dsl

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType
import kotlin.random.Random

/**
 * Image processing DSL extensions for creating BCHW format tensors
 */

/**
 * Creates image tensors in BCHW (Batch, Channel, Height, Width) format
 */
public class ImageTensor<T : DType, V>(private val builder: TensorBuilder<T, V>) {
    
    /**
     * Creates a 4D tensor for image batch in BCHW format
     * @param batchSize number of images in batch
     * @param channels number of channels (e.g., 3 for RGB, 1 for grayscale)
     * @param height image height in pixels
     * @param width image width in pixels
     */
    public fun bchw(batchSize: Int, channels: Int, height: Int, width: Int): ShapeBuilder<T, V> {
        return builder.shape(batchSize, channels, height, width)
    }
    
    /**
     * Creates a single image tensor in CHW format
     * @param channels number of channels (e.g., 3 for RGB, 1 for grayscale)
     * @param height image height in pixels
     * @param width image width in pixels
     */
    public fun chw(channels: Int, height: Int, width: Int): ShapeBuilder<T, V> {
        return builder.shape(channels, height, width)
    }
    
    /**
     * Creates a grayscale image tensor (single channel)
     * @param height image height in pixels
     * @param width image width in pixels
     */
    public fun grayscale(height: Int, width: Int): ShapeBuilder<T, V> {
        return builder.shape(1, height, width)
    }
    
    /**
     * Creates an RGB image tensor (3 channels)
     * @param height image height in pixels
     * @param width image width in pixels
     */
    public fun rgb(height: Int, width: Int): ShapeBuilder<T, V> {
        return builder.shape(3, height, width)
    }
    
    /**
     * Creates an RGBA image tensor (4 channels)
     * @param height image height in pixels
     * @param width image width in pixels
     */
    public fun rgba(height: Int, width: Int): ShapeBuilder<T, V> {
        return builder.shape(4, height, width)
    }
}

/**
 * Creates convolution kernel tensors
 */
public class ConvKernel<T : DType, V>(private val builder: TensorBuilder<T, V>) {
    
    /**
     * Creates a 4D convolution kernel in OIHW format (Output, Input, Height, Width)
     * @param outputChannels number of output channels
     * @param inputChannels number of input channels
     * @param kernelHeight kernel height
     * @param kernelWidth kernel width
     */
    public fun oihw(
        outputChannels: Int, 
        inputChannels: Int, 
        kernelHeight: Int, 
        kernelWidth: Int
    ): ShapeBuilder<T, V> {
        return builder.shape(outputChannels, inputChannels, kernelHeight, kernelWidth)
    }
    
    /**
     * Creates a square convolution kernel
     * @param outputChannels number of output channels
     * @param inputChannels number of input channels
     * @param kernelSize size of the square kernel (both height and width)
     */
    public fun square(
        outputChannels: Int, 
        inputChannels: Int, 
        kernelSize: Int
    ): ShapeBuilder<T, V> {
        return builder.shape(outputChannels, inputChannels, kernelSize, kernelSize)
    }
    
    /**
     * Creates a 1x1 convolution kernel (pointwise convolution)
     * @param outputChannels number of output channels
     * @param inputChannels number of input channels
     */
    public fun pointwise(outputChannels: Int, inputChannels: Int): ShapeBuilder<T, V> {
        return builder.shape(outputChannels, inputChannels, 1, 1)
    }
    
    /**
     * Creates a 3x3 convolution kernel (most common)
     * @param outputChannels number of output channels
     * @param inputChannels number of input channels
     */
    public fun conv3x3(outputChannels: Int, inputChannels: Int): ShapeBuilder<T, V> {
        return builder.shape(outputChannels, inputChannels, 3, 3)
    }
    
    /**
     * Creates a 5x5 convolution kernel
     * @param outputChannels number of output channels
     * @param inputChannels number of input channels
     */
    public fun conv5x5(outputChannels: Int, inputChannels: Int): ShapeBuilder<T, V> {
        return builder.shape(outputChannels, inputChannels, 5, 5)
    }
}

/**
 * Common image processing initializations
 */
public class ImageInitializers<T : DType, V>(private val shapeBuilder: ShapeBuilder<T, V>) {
    
    /**
     * Initialize with pixel values normalized to [0, 1] range
     */
    public fun normalizedPixels(random: Random = Random.Default): TensorInitializer<T, V> {
        return shapeBuilder.uniform(0.0f, 1.0f, random)
    }
    
    /**
     * Initialize with pixel values in standard [0, 255] range
     */
    public fun pixelRange(random: Random = Random.Default): TensorInitializer<T, V> {
        return shapeBuilder.uniform(0.0f, 255.0f, random)
    }
    
    /**
     * Initialize with ImageNet normalization values
     */
    public fun imagenetNormalized(random: Random = Random.Default): TensorInitializer<T, V> {
        // ImageNet mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]
        return shapeBuilder.randn(0.0f, 0.3f, random)
    }
    
    /**
     * Initialize as black image (all zeros)
     */
    public fun black(): TensorInitializer<T, V> {
        return shapeBuilder.zeros()
    }
    
    /**
     * Initialize as white image (all ones for normalized, 255 for pixel range)
     */
    public fun white(): TensorInitializer<T, V> {
        return shapeBuilder.ones()
    }
}

/**
 * Extension function to access image tensor creation
 */
public fun <T : DType, V> TensorBuilder<T, V>.image(): ImageTensor<T, V> = ImageTensor(this)

/**
 * Extension function to access convolution kernel creation
 */
public fun <T : DType, V> TensorBuilder<T, V>.kernel(): ConvKernel<T, V> = ConvKernel(this)

/**
 * Extension function to access image-specific initializers
 */
public fun <T : DType, V> ShapeBuilder<T, V>.imageInit(): ImageInitializers<T, V> = ImageInitializers(this)