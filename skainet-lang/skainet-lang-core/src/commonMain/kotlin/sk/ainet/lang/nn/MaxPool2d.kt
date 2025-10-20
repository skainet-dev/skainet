package sk.ainet.lang.nn

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.DType

/**
 * 2D Max Pooling layer that applies a max pooling operation over 2D input.
 * 
 * Max pooling reduces the spatial dimensions of the input by taking the maximum
 * value in each pooling window.
 * 
 * @param kernelSize Size of the pooling window (height, width)
 * @param stride Stride of the pooling operation (default: same as kernelSize)
 * @param padding Padding added to all sides of the input (default: 0, 0)
 * @param name Name of the module
 */
public class MaxPool2d<T : DType, V>(
    public val kernelSize: Pair<Int, Int>,
    public val stride: Pair<Int, Int> = kernelSize,
    public val padding: Pair<Int, Int> = 0 to 0,
    override val name: String = "MaxPool2d",
) : Module<T, V>() {
    
    init {
        require(kernelSize.first > 0 && kernelSize.second > 0) { "kernelSize must be positive" }
        require(stride.first > 0 && stride.second > 0) { "stride must be positive" }
        require(padding.first >= 0 && padding.second >= 0) { "padding must be non-negative" }
    }

    override val modules: List<Module<T, V>>
        get() = emptyList()

    override fun forward(input: Tensor<T, V>): Tensor<T, V> {
        // For model summary generation, we only need to calculate the correct output shape
        // The actual max pooling computation is not needed for shape inference
        
        val inputShape = input.shape
        require(inputShape.rank == 4) { "MaxPool2d expects 4D input tensor (batch, channels, height, width)" }
        
        val batch = inputShape.dimensions[0]
        val channels = inputShape.dimensions[1] 
        val inputHeight = inputShape.dimensions[2]
        val inputWidth = inputShape.dimensions[3]
        
        val (outputHeight, outputWidth) = outputSize(inputHeight to inputWidth)
        val outputShape = Shape(batch, channels, outputHeight, outputWidth)
        
        // Create a VoidOpsTensor with the correct output shape
        val dataFactory = DenseTensorDataFactory()
        val outputData = dataFactory.zeros<T, V>(outputShape, input.dtype)
        return VoidOpsTensor(outputData, input.dtype)
    }

    /**
     * Calculates the output size for a given input size and pooling parameters.
     */
    public fun outputSize(inputSize: Pair<Int, Int>): Pair<Int, Int> {
        val (inputHeight, inputWidth) = inputSize
        val (kernelHeight, kernelWidth) = kernelSize
        val (strideHeight, strideWidth) = stride
        val (padHeight, padWidth) = padding
        
        val outputHeight = ((inputHeight + 2 * padHeight - kernelHeight) / strideHeight) + 1
        val outputWidth = ((inputWidth + 2 * padWidth - kernelWidth) / strideWidth) + 1
        
        return outputHeight to outputWidth
    }
}