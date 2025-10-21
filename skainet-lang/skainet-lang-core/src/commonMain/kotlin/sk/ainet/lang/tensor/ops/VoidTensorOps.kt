package sk.ainet.lang.tensor.ops

import sk.ainet.lang.ops.InProgress
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.DType

public class VoidTensorOps<V> : TensorOps<V> {
    
    private val dataFactory = DenseTensorDataFactory()
    
    /**
     * Validates that two shapes are compatible for element-wise operations.
     * Implements NumPy-style broadcasting rules.
     */
    private fun validateElementWiseShapes(a: Shape, b: Shape, operation: String) {
        if (!areShapesBroadcastable(a, b)) {
            throw IllegalArgumentException(
                "Shape mismatch for $operation: ${a.dimensions.contentToString()} vs ${b.dimensions.contentToString()}"
            )
        }
    }
    
    /**
     * Checks if two shapes are broadcastable according to NumPy broadcasting rules.
     * Two shapes are broadcastable if:
     * 1. They have the same number of dimensions, or
     * 2. One can be broadcast to match the other by prepending 1s to the smaller shape
     * 3. For each dimension, the sizes are equal or one of them is 1
     */
    private fun areShapesBroadcastable(a: Shape, b: Shape): Boolean {
        val aDims = a.dimensions
        val bDims = b.dimensions
        val maxLen = maxOf(aDims.size, bDims.size)
        
        // Pad shorter shape with 1s at the beginning
        val aPadded = IntArray(maxLen) { i -> 
            if (i < maxLen - aDims.size) 1 else aDims[i - (maxLen - aDims.size)] 
        }
        val bPadded = IntArray(maxLen) { i -> 
            if (i < maxLen - bDims.size) 1 else bDims[i - (maxLen - bDims.size)] 
        }
        
        // Check broadcasting compatibility for each dimension
        for (i in 0 until maxLen) {
            if (aPadded[i] != bPadded[i] && aPadded[i] != 1 && bPadded[i] != 1) {
                return false
            }
        }
        return true
    }
    
    /**
     * Calculates the result shape after broadcasting two shapes.
     * The result shape has the maximum size for each dimension.
     */
    private fun calculateBroadcastShape(a: Shape, b: Shape): Shape {
        val aDims = a.dimensions
        val bDims = b.dimensions
        val maxLen = maxOf(aDims.size, bDims.size)
        
        // Pad shorter shape with 1s at the beginning
        val aPadded = IntArray(maxLen) { i -> 
            if (i < maxLen - aDims.size) 1 else aDims[i - (maxLen - aDims.size)] 
        }
        val bPadded = IntArray(maxLen) { i -> 
            if (i < maxLen - bDims.size) 1 else bDims[i - (maxLen - bDims.size)] 
        }
        
        // Result shape takes the maximum of each dimension
        val resultDims = IntArray(maxLen) { i -> maxOf(aPadded[i], bPadded[i]) }
        return Shape(resultDims)
    }
    
    override fun <T : DType> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        validateElementWiseShapes(a.shape, b.shape, "addition")
        val resultShape = calculateBroadcastShape(a.shape, b.shape)
        val resultData = dataFactory.zeros<T, V>(resultShape, a.dtype)
        return VoidOpsTensor(resultData, a.dtype)
    }

    override fun <T : DType> subtract(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        validateElementWiseShapes(a.shape, b.shape, "subtraction")
        val resultShape = calculateBroadcastShape(a.shape, b.shape)
        val resultData = dataFactory.zeros<T, V>(resultShape, a.dtype)
        return VoidOpsTensor(resultData, a.dtype)
    }

    override fun <T : DType> multiply(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        validateElementWiseShapes(a.shape, b.shape, "multiplication")
        val resultShape = calculateBroadcastShape(a.shape, b.shape)
        val resultData = dataFactory.zeros<T, V>(resultShape, a.dtype)
        return VoidOpsTensor(resultData, a.dtype)
    }

    override fun <T : DType> divide(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        validateElementWiseShapes(a.shape, b.shape, "division")
        val resultShape = calculateBroadcastShape(a.shape, b.shape)
        val resultData = dataFactory.zeros<T, V>(resultShape, a.dtype)
        return VoidOpsTensor(resultData, a.dtype)
    }

    @InProgress("Metal", owner="ops-team", issue="GH-1234")
    override fun <T : DType> matmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        validateMatmulShapes(a.shape, b.shape)
        val resultShape = calculateMatmulShape(a.shape, b.shape)
        val resultData = dataFactory.zeros<T, V>(resultShape, a.dtype)
        return VoidOpsTensor(resultData, a.dtype)
    }

    @InProgress("Metal", owner="ops-team", issue="GH-1234")
    override fun <T : DType> transpose(tensor: Tensor<T, V>): Tensor<T, V> {
        val resultShape = calculateTransposeShape(tensor.shape)
        val resultData = dataFactory.zeros<T, V>(resultShape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType> conv2d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        dilation: Pair<Int, Int>,
        groups: Int
    ): Tensor<T, V> {
        val resultShape = calculateConv2dShape(input.shape, weight.shape, stride, padding, dilation)
        val resultData = dataFactory.zeros<T, V>(resultShape, input.dtype)
        return VoidOpsTensor(resultData, input.dtype)
    }

    override fun <T : DType> maxPool2d(
        input: Tensor<T, V>,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>
    ): Tensor<T, V> {
        val resultShape = calculateMaxPool2dShape(input.shape, kernelSize, stride, padding)
        val resultData = dataFactory.zeros<T, V>(resultShape, input.dtype)
        return VoidOpsTensor(resultData, input.dtype)
    }

    override fun <T : DType> reshape(tensor: Tensor<T, V>, newShape: Shape): Tensor<T, V> {
        validateReshape(tensor.shape, newShape)
        val resultData = dataFactory.zeros<T, V>(newShape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType> flatten(tensor: Tensor<T, V>, startDim: Int, endDim: Int): Tensor<T, V> {
        val resultShape = calculateFlattenShape(tensor.shape, startDim, endDim)
        val resultData = dataFactory.zeros<T, V>(resultShape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType> concat(tensors: List<Tensor<T, V>>, dim: Int): Tensor<T, V> {
        val resultShape = calculateConcatShape(tensors.map { it.shape }, dim)
        val resultData = dataFactory.zeros<T, V>(resultShape, tensors.first().dtype)
        return VoidOpsTensor(resultData, tensors.first().dtype)
    }

    override fun <T : DType> split(tensor: Tensor<T, V>, splitSize: Int, dim: Int): List<Tensor<T, V>> {
        val resultShapes = calculateSplitShapes(tensor.shape, splitSize, dim)
        return resultShapes.map { shape ->
            val resultData = dataFactory.zeros<T, V>(shape, tensor.dtype)
            VoidOpsTensor(resultData, tensor.dtype)
        }
    }

    override fun <T : DType> squeeze(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> {
        val resultShape = calculateSqueezeShape(tensor.shape, dim)
        val resultData = dataFactory.zeros<T, V>(resultShape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType> unsqueeze(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> {
        val resultShape = calculateUnsqueezeShape(tensor.shape, dim)
        val resultData = dataFactory.zeros<T, V>(resultShape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType> relu(tensor: Tensor<T, V>): Tensor<T, V> {
        // Activation functions preserve shape
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType> softmax(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> {
        validateSoftmaxDim(tensor.shape, dim)
        // Softmax preserves shape
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType> sigmoid(tensor: Tensor<T, V>): Tensor<T, V> {
        // Activation functions preserve shape
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType> sum(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> {
        val resultShape = calculateReductionShape(tensor.shape, dim, "sum")
        val resultData = dataFactory.zeros<T, V>(resultShape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType> mean(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> {
        val resultShape = calculateReductionShape(tensor.shape, dim, "mean")
        val resultData = dataFactory.zeros<T, V>(resultShape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <TFrom : DType, TTo : DType> convert(
        tensor: Tensor<TFrom, V>,
        targetType: TTo
    ): Tensor<TTo, V> {
        // Type conversion preserves shape but changes dtype
        @Suppress("UNCHECKED_CAST")
        val targetClass = targetType::class as kotlin.reflect.KClass<TTo>
        val resultData = dataFactory.zeros<TTo, V>(tensor.shape, targetClass)
        return VoidOpsTensor(resultData, targetClass)
    }

    override fun <T : DType> silu(tensor: Tensor<T, V>): Tensor<T, V> {
        // SiLU (Swish) activation function preserves shape
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType> gelu(tensor: Tensor<T, V>): Tensor<T, V> {
        // GELU activation function preserves shape
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType> variance(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> {
        val resultShape = calculateReductionShape(tensor.shape, dim, "variance")
        val resultData = dataFactory.zeros<T, V>(resultShape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType> sqrt(tensor: Tensor<T, V>): Tensor<T, V> {
        // Square root function preserves shape
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    /**
     * Validates shapes for matrix multiplication.
     * For 2D matrices: (m, k) × (k, n) -> (m, n)
     * For higher dimensions: batch dimensions must match, inner dimensions must be compatible
     */
    private fun validateMatmulShapes(a: Shape, b: Shape) {
        if (a.rank < 2 || b.rank < 2) {
            throw IllegalArgumentException("Matrix multiplication requires tensors with at least 2 dimensions")
        }
        
        val aLastDim = a.dimensions[a.rank - 1]
        val bSecondLastDim = b.dimensions[b.rank - 2]
        
        if (aLastDim != bSecondLastDim) {
            throw IllegalArgumentException(
                "Matrix multiplication shape mismatch: inner dimensions must match " +
                "($aLastDim vs $bSecondLastDim)"
            )
        }
        
        // For tensors with more than 2 dimensions, batch dimensions must be compatible
        if (a.rank > 2 || b.rank > 2) {
            val maxRank = maxOf(a.rank, b.rank)
            for (i in 0 until maxRank - 2) {
                val aDim = if (i < a.rank - 2) a.dimensions[i] else 1
                val bDim = if (i < b.rank - 2) b.dimensions[i] else 1
                if (aDim != bDim && aDim != 1 && bDim != 1) {
                    throw IllegalArgumentException(
                        "Matrix multiplication batch dimension mismatch at position $i: $aDim vs $bDim"
                    )
                }
            }
        }
    }

    /**
     * Calculates the result shape for matrix multiplication
     */
    private fun calculateMatmulShape(a: Shape, b: Shape): Shape {
        val maxRank = maxOf(a.rank, b.rank)
        val resultDims = IntArray(maxRank)
        
        // Handle batch dimensions
        for (i in 0 until maxRank - 2) {
            val aDim = if (i < a.rank - 2) a.dimensions[i] else 1
            val bDim = if (i < b.rank - 2) b.dimensions[i] else 1
            resultDims[i] = maxOf(aDim, bDim)
        }
        
        // Handle matrix dimensions: (m, k) × (k, n) -> (m, n)
        resultDims[maxRank - 2] = a.dimensions[a.rank - 2]
        resultDims[maxRank - 1] = b.dimensions[b.rank - 1]
        
        return Shape(resultDims)
    }

    /**
     * Calculates the result shape for transpose operation.
     * For 2D tensors: (m, n) -> (n, m)
     * For higher dimensions: swaps the last two dimensions
     */
    private fun calculateTransposeShape(shape: Shape): Shape {
        if (shape.rank < 2) {
            throw IllegalArgumentException("Transpose requires tensors with at least 2 dimensions")
        }
        
        val resultDims = shape.dimensions.copyOf()
        val lastIdx = resultDims.size - 1
        val secondLastIdx = resultDims.size - 2
        
        // Swap last two dimensions
        val temp = resultDims[lastIdx]
        resultDims[lastIdx] = resultDims[secondLastIdx]
        resultDims[secondLastIdx] = temp
        
        return Shape(resultDims)
    }

    /**
     * Validates reshape operation - total volume must remain the same
     */
    private fun validateReshape(originalShape: Shape, newShape: Shape) {
        if (originalShape.volume != newShape.volume) {
            throw IllegalArgumentException(
                "Reshape volume mismatch: original volume ${originalShape.volume} != " +
                "new volume ${newShape.volume}"
            )
        }
    }

    /**
     * Calculates the result shape for flatten operation
     */
    private fun calculateFlattenShape(shape: Shape, startDim: Int, endDim: Int): Shape {
        val actualStartDim = if (startDim < 0) shape.rank + startDim else startDim
        val actualEndDim = if (endDim < 0) shape.rank + endDim else endDim
        
        if (actualStartDim < 0 || actualStartDim >= shape.rank) {
            throw IllegalArgumentException("Start dimension $startDim is out of bounds for tensor with ${shape.rank} dimensions")
        }
        if (actualEndDim < 0 || actualEndDim >= shape.rank) {
            throw IllegalArgumentException("End dimension $endDim is out of bounds for tensor with ${shape.rank} dimensions")
        }
        if (actualStartDim > actualEndDim) {
            throw IllegalArgumentException("Start dimension $actualStartDim must be <= end dimension $actualEndDim")
        }
        
        val resultDims = mutableListOf<Int>()
        
        // Add dimensions before startDim
        for (i in 0 until actualStartDim) {
            resultDims.add(shape.dimensions[i])
        }
        
        // Calculate flattened dimension
        var flattenedSize = 1
        for (i in actualStartDim..actualEndDim) {
            flattenedSize *= shape.dimensions[i]
        }
        resultDims.add(flattenedSize)
        
        // Add dimensions after endDim
        for (i in actualEndDim + 1 until shape.rank) {
            resultDims.add(shape.dimensions[i])
        }
        
        return Shape(resultDims.toIntArray())
    }

    /**
     * Validates softmax dimension parameter
     */
    private fun validateSoftmaxDim(shape: Shape, dim: Int) {
        val actualDim = if (dim < 0) shape.rank + dim else dim
        if (actualDim < 0 || actualDim >= shape.rank) {
            throw IllegalArgumentException("Softmax dimension $dim is out of bounds for tensor with ${shape.rank} dimensions")
        }
    }

    /**
     * Calculates the result shape for reduction operations (sum, mean)
     */
    private fun calculateReductionShape(shape: Shape, dim: Int?, operation: String): Shape {
        return if (dim == null) {
            // Reduce all dimensions to scalar
            Shape(1)
        } else {
            val actualDim = if (dim < 0) shape.rank + dim else dim
            if (actualDim < 0 || actualDim >= shape.rank) {
                throw IllegalArgumentException("$operation dimension $dim is out of bounds for tensor with ${shape.rank} dimensions")
            }
            
            // Remove the specified dimension
            val resultDims = shape.dimensions.filterIndexed { index, _ -> index != actualDim }.toIntArray()
            if (resultDims.isEmpty()) {
                Shape(1) // Result is scalar if all dimensions are reduced
            } else {
                Shape(resultDims)
            }
        }
    }

    /**
     * Calculates the result shape for conv2d operation.
     * Input shape: (batch, in_channels, height, width)
     * Weight shape: (out_channels, in_channels_per_group, kernel_height, kernel_width)
     * Output shape: (batch, out_channels, out_height, out_width)
     */
    private fun calculateConv2dShape(
        inputShape: Shape, 
        weightShape: Shape, 
        stride: Pair<Int, Int>, 
        padding: Pair<Int, Int>, 
        dilation: Pair<Int, Int>
    ): Shape {
        if (inputShape.rank != 4) {
            throw IllegalArgumentException("Conv2d input must be 4D tensor (batch, channels, height, width)")
        }
        if (weightShape.rank != 4) {
            throw IllegalArgumentException("Conv2d weight must be 4D tensor (out_channels, in_channels, kernel_h, kernel_w)")
        }
        
        val batch = inputShape.dimensions[0]
        val outChannels = weightShape.dimensions[0]
        val inputHeight = inputShape.dimensions[2]
        val inputWidth = inputShape.dimensions[3]
        val kernelHeight = weightShape.dimensions[2]
        val kernelWidth = weightShape.dimensions[3]
        
        val (strideH, strideW) = stride
        val (padH, padW) = padding
        val (dilationH, dilationW) = dilation
        
        val outputHeight = ((inputHeight + 2 * padH - dilationH * (kernelHeight - 1) - 1) / strideH) + 1
        val outputWidth = ((inputWidth + 2 * padW - dilationW * (kernelWidth - 1) - 1) / strideW) + 1
        
        return Shape(batch, outChannels, outputHeight, outputWidth)
    }

    /**
     * Calculates the result shape for maxPool2d operation.
     * Input shape: (batch, channels, height, width)
     * Output shape: (batch, channels, out_height, out_width)
     */
    private fun calculateMaxPool2dShape(
        inputShape: Shape,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>
    ): Shape {
        if (inputShape.rank != 4) {
            throw IllegalArgumentException("MaxPool2d input must be 4D tensor (batch, channels, height, width)")
        }
        
        val batch = inputShape.dimensions[0]
        val channels = inputShape.dimensions[1]
        val inputHeight = inputShape.dimensions[2]
        val inputWidth = inputShape.dimensions[3]
        
        val (kernelH, kernelW) = kernelSize
        val (strideH, strideW) = stride
        val (padH, padW) = padding
        
        val outputHeight = ((inputHeight + 2 * padH - kernelH) / strideH) + 1
        val outputWidth = ((inputWidth + 2 * padW - kernelW) / strideW) + 1
        
        return Shape(batch, channels, outputHeight, outputWidth)
    }

    /**
     * Calculates the result shape for concat operation
     */
    private fun calculateConcatShape(shapes: List<Shape>, dim: Int): Shape {
        if (shapes.isEmpty()) {
            throw IllegalArgumentException("Cannot concatenate empty list of tensors")
        }
        
        val firstShape = shapes.first()
        val actualDim = if (dim < 0) firstShape.rank + dim else dim
        
        if (actualDim < 0 || actualDim >= firstShape.rank) {
            throw IllegalArgumentException("Concatenation dimension $dim is out of bounds for tensor with ${firstShape.rank} dimensions")
        }
        
        // Validate all shapes are compatible (same except in concat dimension)
        for (shape in shapes.drop(1)) {
            if (shape.rank != firstShape.rank) {
                throw IllegalArgumentException("All tensors must have the same number of dimensions for concatenation")
            }
            for (i in shape.dimensions.indices) {
                if (i != actualDim && shape.dimensions[i] != firstShape.dimensions[i]) {
                    throw IllegalArgumentException(
                        "All tensors must have the same shape except in the concatenation dimension. " +
                        "Dimension $i: ${firstShape.dimensions[i]} vs ${shape.dimensions[i]}"
                    )
                }
            }
        }
        
        // Calculate result shape
        val resultDims = firstShape.dimensions.copyOf()
        resultDims[actualDim] = shapes.sumOf { it.dimensions[actualDim] }
        
        return Shape(resultDims)
    }

    /**
     * Calculates the result shapes for split operation
     */
    private fun calculateSplitShapes(shape: Shape, splitSize: Int, dim: Int): List<Shape> {
        val actualDim = if (dim < 0) shape.rank + dim else dim
        
        if (actualDim < 0 || actualDim >= shape.rank) {
            throw IllegalArgumentException("Split dimension $dim is out of bounds for tensor with ${shape.rank} dimensions")
        }
        
        if (splitSize <= 0) {
            throw IllegalArgumentException("Split size must be positive, got $splitSize")
        }
        
        val dimSize = shape.dimensions[actualDim]
        if (dimSize % splitSize != 0) {
            throw IllegalArgumentException(
                "Tensor dimension $dimSize is not divisible by split size $splitSize"
            )
        }
        
        val numSplits = dimSize / splitSize
        val resultDims = shape.dimensions.copyOf()
        resultDims[actualDim] = splitSize
        
        return List(numSplits) { Shape(resultDims) }
    }

    /**
     * Calculates the result shape for squeeze operation
     */
    private fun calculateSqueezeShape(shape: Shape, dim: Int?): Shape {
        return if (dim == null) {
            // Remove all dimensions of size 1
            val resultDims = shape.dimensions.filter { it != 1 }.toIntArray()
            if (resultDims.isEmpty()) {
                Shape(1) // If all dimensions were 1, result is scalar
            } else {
                Shape(resultDims)
            }
        } else {
            val actualDim = if (dim < 0) shape.rank + dim else dim
            
            if (actualDim < 0 || actualDim >= shape.rank) {
                throw IllegalArgumentException("Squeeze dimension $dim is out of bounds for tensor with ${shape.rank} dimensions")
            }
            
            if (shape.dimensions[actualDim] != 1) {
                throw IllegalArgumentException(
                    "Cannot squeeze dimension $actualDim with size ${shape.dimensions[actualDim]}. Only dimensions of size 1 can be squeezed."
                )
            }
            
            // Remove the specified dimension
            val resultDims = shape.dimensions.filterIndexed { index, _ -> index != actualDim }.toIntArray()
            if (resultDims.isEmpty()) {
                Shape(1) // Result is scalar if all dimensions are removed
            } else {
                Shape(resultDims)
            }
        }
    }

    /**
     * Calculates the result shape for unsqueeze operation
     */
    private fun calculateUnsqueezeShape(shape: Shape, dim: Int): Shape {
        val newRank = shape.rank + 1
        val actualDim = if (dim < 0) newRank + dim else dim
        
        if (actualDim < 0 || actualDim >= newRank) {
            throw IllegalArgumentException("Unsqueeze dimension $dim is out of bounds for new tensor with $newRank dimensions")
        }
        
        val resultDims = IntArray(newRank)
        var originalIndex = 0
        
        for (i in 0 until newRank) {
            if (i == actualDim) {
                resultDims[i] = 1
            } else {
                resultDims[i] = shape.dimensions[originalIndex]
                originalIndex++
            }
        }
        
        return Shape(resultDims)
    }
}
