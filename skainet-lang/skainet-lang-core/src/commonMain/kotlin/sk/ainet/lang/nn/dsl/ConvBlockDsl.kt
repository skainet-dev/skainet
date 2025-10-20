package sk.ainet.lang.nn.dsl

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.types.DType

/**
 * DSL interface for building convolutional blocks with specific precision types.
 * This interface enables mixed-precision neural networks by allowing convolutional 
 * layers to use different precision than the network default.
 * 
 * The block-level precision (TBlock) serves as the default for all layers within
 * the block, but individual layers can override this with their own precision.
 * 
 * @param TBlock The default precision type for this convolutional block
 * @param V The value type that corresponds to the native Kotlin type for the DType
 * 
 * Example usage:
 * ```kotlin
 * network<FP32, Float> {
 *     convBlock<FP16> {  // Block uses FP16 precision
 *         conv2d(32) { /* inherits FP16 from block */ }
 *         batchNorm<FP32> { /* overrides to FP32 */ }
 *         relu()
 *     }
 * }
 * ```
 */
/*
@NetworkDsl
public interface ConvBlockDsl<TBlock : DType, V> : NetworkDslItem {
    
    /**
     * Creates a 2D convolutional layer with the block's default precision.
     * 
     * @param filters Number of output filters/channels
     * @param content Configuration block for layer-specific parameters (kernel size, stride, etc.)
     */
    public fun conv2d(filters: Int, content: CONV2D<TBlock, V>.() -> Unit = {})
    
    /**
     * Creates a batch normalization layer with optional precision override.
     * 
     * @param TLayer The precision type for this specific layer (defaults to TBlock if not specified)
     * @param content Configuration block for batch normalization parameters
     */
    public fun <TLayer : DType> batchNorm(
        content: BATCH_NORM<TLayer, V>.() -> Unit = {}
    )
    
    /**
     * Creates a ReLU activation layer.
     * Activation functions typically don't change precision, they operate in-place.
     */
    public fun relu()
    
    /**
     * Creates a max pooling layer.
     * 
     * @param poolSize Size of the pooling window
     * @param stride Stride of the pooling operation
     * @param content Configuration block for additional pooling parameters
     */
    public fun maxPool2d(
        poolSize: Int = 2, 
        stride: Int = poolSize,
        content: MAX_POOL_2D<TBlock, V>.() -> Unit = {}
    )
    
    /**
     * Creates a dropout layer for regularization.
     * 
     * @param rate Dropout rate (probability of setting elements to zero)
     * @param content Configuration block for dropout parameters
     */
    public fun dropout(rate: Double = 0.5, content: DROPOUT<TBlock, V>.() -> Unit = {})
    
    /**
     * Applies a custom activation function as a separate layer.
     * 
     * @param activation Function that transforms tensor values
     */
    public fun activation(activation: (Tensor<TBlock, V>) -> Tensor<TBlock, V>)
    
    /**
     * Creates a sequential sub-block within this convolutional block.
     * Useful for grouping related layers together.
     * 
     * @param content DSL block containing the sequence of layers
     */
    public fun sequential(content: ConvBlockDsl<TBlock, V>.() -> Unit)
}

/**
 * Configuration interface for 2D convolutional layers.
 * Provides access to convolution-specific parameters and tensor factories.
 * 
 * @param T The precision type for this convolutional layer
 * @param V The value type corresponding to the DType
 */
@NetworkDsl
public interface CONV2D<T : DType, V> : NetworkDslItem {
    /**
     * Number of input channels (automatically inferred from previous layer)
     */
    public var inputChannels: Int
    
    /**
     * Number of output filters/channels
     */
    public var outputChannels: Int
    
    /**
     * Kernel/filter size (height, width)
     */
    public var kernelSize: Pair<Int, Int>
    
    /**
     * Stride for the convolution operation
     */
    public var stride: Pair<Int, Int>
    
    /**
     * Padding applied to the input
     */
    public var padding: Pair<Int, Int>
    
    /**
     * Dilation rate for dilated convolution
     */
    public var dilation: Pair<Int, Int>
    
    /**
     * Number of groups for grouped convolution
     */
    public var groups: Int
    
    /**
     * Whether to include bias parameters
     */
    public var useBias: Boolean
    
    /**
     * Tensor factory for creating weights and biases with correct precision
     */
    public val factory: TensorDataFactory<T, V>
    
    /**
     * Configure kernel weights initialization.
     * 
     * @param initBlock Initialization function that receives the weight shape
     */
    public fun weights(initBlock: WeightsScope<T, V>.(shape: sk.ainet.lang.tensor.Shape) -> Tensor<T, V>)
    
    /**
     * Configure bias initialization (if useBias is true).
     * 
     * @param initBlock Initialization function that receives the bias shape
     */
    public fun bias(initBlock: BiasScope<T, V>.(shape: sk.ainet.lang.tensor.Shape) -> Tensor<T, V>)
}

/**
 * Configuration interface for batch normalization layers.
 * 
 * @param T The precision type for this batch normalization layer
 * @param V The value type corresponding to the DType
 */
@NetworkDsl
public interface BATCH_NORM<T : DType, V> : NetworkDslItem {
    /**
     * Number of features/channels to normalize
     */
    public var numFeatures: Int
    
    /**
     * Small value added to variance for numerical stability
     */
    public var eps: Double
    
    /**
     * Momentum for running statistics update
     */
    public var momentum: Double
    
    /**
     * Whether to learn affine transformation parameters (scale and shift)
     */
    public var affine: Boolean
    
    /**
     * Whether to track running statistics for inference
     */
    public var trackRunningStats: Boolean
    
    /**
     * Tensor factory for creating scale and shift parameters
     */
    public val factory: TensorDataFactory<T, V>
}

/**
 * Configuration interface for 2D max pooling layers.
 * 
 * @param T The precision type for this pooling layer
 * @param V The value type corresponding to the DType
 */
@NetworkDsl
public interface MAX_POOL_2D<T : DType, V> : NetworkDslItem {
    /**
     * Size of the pooling window
     */
    public var poolSize: Pair<Int, Int>
    
    /**
     * Stride of the pooling operation
     */
    public var stride: Pair<Int, Int>
    
    /**
     * Padding applied before pooling
     */
    public var padding: Pair<Int, Int>
    
    /**
     * Dilation for dilated pooling
     */
    public var dilation: Pair<Int, Int>
    
    /**
     * Whether to return indices of maximum values
     */
    public var returnIndices: Boolean
    
    /**
     * Whether to use ceil mode for output size calculation
     */
    public var ceilMode: Boolean
}


 */