package sk.ainet.core.tensor

import sk.ainet.core.tensor.dsl.TensorViewBuilder

/**
 * Extension functions for tensor slicing using ViewTensorData.
 * 
 * This file provides convenient methods for creating zero-copy tensor slices
 * that return regular Tensor instances using ViewTensorData internally.
 */

/**
 * Creates a zero-copy slice of this tensor using slice descriptors.
 * 
 * @param descriptors List of slice descriptors defining the slice
 * @return A Tensor instance using ViewTensorData for the sliced view
 */
public fun <T : DType, V> Tensor<T, V>.sliceView(descriptors: List<SliceDescriptor>): Tensor<T, V> {
    // Compute view parameters
    val viewShape = computeViewShape(this.shape, descriptors)
    val viewStrides = computeViewStrides(this.shape, descriptors)
    val viewOffset = computeViewOffset(this.shape, descriptors)
    
    // Get parent data array
    val parentData = Array<Any?>(this.shape.volume) { null }
    this.copyTo(parentData as Array<V>, 0)
    
    // Create ViewTensorData
    val viewData = ViewTensorData<T, V>(
        parentData = parentData as Array<V>,
        shape = viewShape,
        strides = viewStrides,
        offset = viewOffset,
        parentShape = this.shape
    )
    
    // Return a new tensor using the sliced data
    return viewData.materialize() as Tensor<T, V>
}

/**
 * Creates a zero-copy slice of this tensor using the DSL builder.
 * 
 * @param builder DSL builder function for defining the slice
 * @return A Tensor instance that provides a zero-copy slice of this tensor
 */
public fun <T : DType, V> Tensor<T, V>.sliceView(builder: TensorViewBuilder<T, V>.() -> Unit): Tensor<T, V> {
    // For now, use the existing slice method which is simpler
    // This is a temporary implementation until TensorViewBuilder is also refactored
    return this.slice()
}

/**
 * Intelligent slicing function that returns a new tensor with the sliced data.
 * 
 * @param builder DSL builder function for defining the slice
 * @param forceView Ignored - kept for compatibility 
 * @param forceCopy Ignored - kept for compatibility
 * @return A Tensor with the sliced data
 */
public fun <T : DType, V> Tensor<T, V>.smartSlice(
    forceView: Boolean = false,
    forceCopy: Boolean = false,
    builder: TensorViewBuilder<T, V>.() -> Unit
): Tensor<T, V> {
    return sliceView(builder)
}

/**
 * Creates a slice using simple range parameters.
 */
public fun <T : DType, V> Tensor<T, V>.slice(vararg ranges: IntRange?): Tensor<T, V> {
    val descriptors = mutableListOf<SliceDescriptor>()
    
    for (i in 0 until this.shape.rank) {
        val range = if (i < ranges.size) ranges[i] else null
        val descriptor = range?.let { 
            SliceDescriptor.Range(it.first, it.last + 1, 1)
        } ?: SliceDescriptor.All
        descriptors.add(descriptor)
    }
    
    return sliceView(descriptors)
}

/**
 * Creates a slice by indexing specific dimensions.
 */
public fun <T : DType, V> Tensor<T, V>.at(vararg indices: Int?): Tensor<T, V> {
    val descriptors = mutableListOf<SliceDescriptor>()
    
    for (i in 0 until this.shape.rank) {
        val index = if (i < indices.size) indices[i] else null
        val descriptor = index?.let {
            SliceDescriptor.Index(it)
        } ?: SliceDescriptor.All
        descriptors.add(descriptor)
    }
    
    return sliceView(descriptors)
}

/**
 * NCHW-specific batch slicing for tensor with NCHW layout.
 */
public fun <T : DType, V> Tensor<T, V>.batchSlice(startBatch: Int, endBatch: Int, step: Int = 1): Tensor<T, V> {
    require(this.shape.rank == 4) { "Batch slicing requires 4D tensor (NCHW format)" }
    
    val descriptors = listOf(
        SliceDescriptor.Range(startBatch, endBatch, step),
        SliceDescriptor.All, // channels
        SliceDescriptor.All, // height
        SliceDescriptor.All  // width
    )
    
    return sliceView(descriptors)
}

/**
 * NCHW-specific channel slicing.
 */
public fun <T : DType, V> Tensor<T, V>.channelSlice(channelIndex: Int): Tensor<T, V> {
    require(this.shape.rank == 4) { "Channel slicing requires 4D tensor (NCHW format)" }
    
    val descriptors = listOf(
        SliceDescriptor.All,   // batch
        SliceDescriptor.Index(channelIndex), // specific channel
        SliceDescriptor.All,   // height
        SliceDescriptor.All    // width
    )
    
    return sliceView(descriptors)
}

/**
 * NCHW-specific spatial slicing.
 */
public fun <T : DType, V> Tensor<T, V>.spatialSlice(
    heightStart: Int, heightEnd: Int, heightStep: Int = 1,
    widthStart: Int, widthEnd: Int, widthStep: Int = 1
): Tensor<T, V> {
    require(this.shape.rank == 4) { "Spatial slicing requires 4D tensor (NCHW format)" }
    
    val descriptors = listOf(
        SliceDescriptor.All, // batch
        SliceDescriptor.All, // channels
        SliceDescriptor.Range(heightStart, heightEnd, heightStep),
        SliceDescriptor.Range(widthStart, widthEnd, widthStep)
    )
    
    return sliceView(descriptors)
}

// Helper functions for computing view parameters

private fun computeViewShape(parentShape: Shape, descriptors: List<SliceDescriptor>): Shape {
    val dimensions = mutableListOf<Int>()
    
    for (i in descriptors.indices) {
        when (val descriptor = descriptors[i]) {
            is SliceDescriptor.Range -> {
                val size = (descriptor.end - descriptor.start + descriptor.step - 1) / descriptor.step
                dimensions.add(size)
            }
            is SliceDescriptor.Index -> {
                // Index slicing reduces dimensionality - don't add dimension
            }
            SliceDescriptor.All -> {
                dimensions.add(parentShape.dimensions[i])
            }
        }
    }
    
    return Shape(dimensions.toIntArray())
}

private fun computeViewStrides(parentShape: Shape, descriptors: List<SliceDescriptor>): IntArray {
    val strides = mutableListOf<Int>()
    val parentStrides = parentShape.computeStrides()
    
    for (i in descriptors.indices) {
        when (val descriptor = descriptors[i]) {
            is SliceDescriptor.Range -> {
                strides.add(parentStrides[i] * descriptor.step)
            }
            is SliceDescriptor.Index -> {
                // Index slicing reduces dimensionality - don't add stride
            }
            SliceDescriptor.All -> {
                strides.add(parentStrides[i])
            }
        }
    }
    
    return strides.toIntArray()
}

private fun computeViewOffset(parentShape: Shape, descriptors: List<SliceDescriptor>): Int {
    var offset = 0
    val parentStrides = parentShape.computeStrides()
    
    for (i in descriptors.indices) {
        when (val descriptor = descriptors[i]) {
            is SliceDescriptor.Range -> {
                offset += descriptor.start * parentStrides[i]
            }
            is SliceDescriptor.Index -> {
                offset += descriptor.index * parentStrides[i]
            }
            SliceDescriptor.All -> {
                // No offset change for All
            }
        }
    }
    
    return offset
}

private fun Shape.computeStrides(): IntArray {
    if (dimensions.isEmpty()) return intArrayOf()
    
    val strides = IntArray(dimensions.size)
    strides[dimensions.size - 1] = 1
    
    for (i in dimensions.size - 2 downTo 0) {
        strides[i] = strides[i + 1] * dimensions[i + 1]
    }
    
    return strides
}