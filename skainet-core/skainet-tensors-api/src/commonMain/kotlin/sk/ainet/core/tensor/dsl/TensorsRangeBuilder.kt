package sk.ainet.core.tensor.dsl

import sk.ainet.core.tensor.DType
import sk.ainet.core.tensor.NCHWViewHelper
import sk.ainet.core.tensor.Slice
import sk.ainet.core.tensor.SliceDescriptor
import sk.ainet.core.tensor.SliceIndexMapper
import sk.ainet.core.tensor.SlicedTensorView
import sk.ainet.core.tensor.Tensor
import sk.ainet.core.tensor.TensorView

/**
 * DSL for tensor slicing mimicking numpy slicing.
 * 
 * Usage:
 * val slicedTensor = slice(tensor) {
 *     segment {
 *         all()
 *     }
 *     segment {
 *         from(1)
 *     }
 * }
 */

/**
 * Main DSL function for slicing tensors
 */
public fun <T : DType, V> slice(tensor: Tensor<T, V>, builder: TensorSliceBuilder<T, V>.() -> Unit): List<Slice<T, V>> {
    val sliceBuilder = TensorSliceBuilder(tensor)
    sliceBuilder.builder()
    return sliceBuilder.build()
}

/**
 * DSL function for creating zero-copy tensor views
 */
public fun <T : DType, V> sliceView(tensor: Tensor<T, V>, builder: TensorViewBuilder<T, V>.() -> Unit): TensorView<T, V> {
    val viewBuilder = TensorViewBuilder(tensor)
    viewBuilder.builder()
    return viewBuilder.buildView()
}

/**
 * DSL function for creating NCHW-optimized tensor views
 */
public fun <T : DType, V> nchwView(tensor: Tensor<T, V>, builder: NCHWViewBuilder<T, V>.() -> Unit): TensorView<T, V> {
    val viewBuilder = NCHWViewBuilder(tensor)
    viewBuilder.builder()
    return viewBuilder.buildView()
}

/**
 * Builder class for tensor slicing DSL
 */
public class TensorSliceBuilder<T : DType, V>(private val tensor: Tensor<T, V>) {
    private val segments = mutableListOf<SegmentBuilder<T, V>>()
    private var currentDimensionIndex = 0

    /**
     * Creates a new segment for the current dimension
     */
    public fun segment(builder: SegmentBuilder<T, V>.() -> Unit) {
        val segmentBuilder = SegmentBuilder(tensor, currentDimensionIndex)
        segmentBuilder.builder()
        segments.add(segmentBuilder)
        currentDimensionIndex++
    }

    /**
     * Builds the list of slices from all segments
     */
    internal fun build(): List<Slice<T, V>> {
        return segments.map { it.build() }
    }
}

/**
 * Builder class for tensor view creation DSL
 */
public class TensorViewBuilder<T : DType, V>(private val tensor: Tensor<T, V>) {
    private val segments = mutableListOf<ViewSegmentBuilder<T, V>>()
    private var currentDimensionIndex = 0
    private var enableFallback = true
    private var complexityThreshold = 10

    /**
     * Creates a new segment for the current dimension
     */
    public fun segment(builder: ViewSegmentBuilder<T, V>.() -> Unit) {
        val segmentBuilder = ViewSegmentBuilder(tensor, currentDimensionIndex)
        segmentBuilder.builder()
        segments.add(segmentBuilder)
        currentDimensionIndex++
    }

    /**
     * Configures the fallback mechanism for view creation.
     * 
     * @param enabled Whether to enable automatic fallback to copying
     * @param threshold Complexity threshold above which fallback is triggered
     */
    public fun fallback(enabled: Boolean = true, threshold: Int = 10) {
        enableFallback = enabled
        complexityThreshold = threshold
    }

    /**
     * Builds the tensor view from all segments with fallback support
     */
    internal fun buildView(): TensorView<T, V> {
        val descriptors = segments.map { it.buildDescriptor() }
        val indexMapper = SliceIndexMapper(tensor.shape, descriptors)
        val view = SlicedTensorView(tensor, indexMapper, descriptors)
        
        // Check if fallback should be applied
        if (enableFallback && shouldFallback()) {
            // Apply materialization fallback - convert complex view to concrete tensor
            val materializedData = view.materialize()
            
            // If materialization produced a concrete tensor, return it as a view
            if (materializedData is TensorView<T, V>) {
                return materializedData
            }
            
            // For cases where materialization returns TensorData but not a full Tensor,
            // we still return the view but with the understanding that it should be
            // materialized when tensor factories become available
            return view
        }
        
        return view
    }

    /**
     * Determines if fallback to copying should be used based on complexity analysis.
     */
    private fun shouldFallback(): Boolean {
        val totalComplexity = segments.sumOf { it.complexityHint }
        val hasNonContiguousSegments = segments.any { !it.isContiguousHint }
        
        // Use fallback if complexity is too high or we have many non-contiguous segments
        return totalComplexity > complexityThreshold || 
               (hasNonContiguousSegments && segments.count { !it.isContiguousHint } > tensor.shape.rank / 2)
    }
}

/**
 * Builder class for NCHW-optimized tensor view creation DSL
 */
public class NCHWViewBuilder<T : DType, V>(private val tensor: Tensor<T, V>) {
    private var batchSlice: IntRange? = null
    private var channelSlice: IntRange? = null
    private var channelIndex: Int? = null
    private var heightSlice: IntRange? = null
    private var widthSlice: IntRange? = null

    init {
        require(tensor.shape.rank == 4) {
            "NCHW views require 4D tensors, got ${tensor.shape.rank}D"
        }
    }

    /**
     * Slice batch dimension: tensor[startBatch:endBatch, :, :, :]
     */
    public fun batch(range: IntRange) {
        batchSlice = range
    }

    /**
     * Slice batch dimension with step: tensor[startBatch:endBatch:step, :, :, :]
     */
    public fun batch(start: Int, end: Int, step: Int = 1) {
        batchSlice = start..end
        // Note: step handling would need to be added to the builder logic
        // TODO add handling would need to be added to the builder logic
    }

    /**
     * Select all channels: tensor[:, :, :, :]
     */
    public fun allChannels() {
        channelSlice = 0 until tensor.shape[1]
    }

    /**
     * Extract specific channel: tensor[:, channelIndex, :, :]
     */
    public fun channel(index: Int) {
        channelIndex = index
    }

    /**
     * Slice channel range: tensor[:, startChannel:endChannel, :, :]
     */
    public fun channels(range: IntRange) {
        channelSlice = range
    }

    /**
     * Slice spatial region: tensor[:, :, heightRange, widthRange]
     */
    public fun spatial(heightRange: IntRange, widthRange: IntRange) {
        heightSlice = heightRange
        widthSlice = widthRange
    }

    /**
     * Slice height dimension: tensor[:, :, heightRange, :]
     */
    public fun height(range: IntRange) {
        heightSlice = range
    }

    /**
     * Slice width dimension: tensor[:, :, :, widthRange]
     */
    public fun width(range: IntRange) {
        widthSlice = range
    }

    /**
     * Builds the NCHW-optimized tensor view
     */
    internal fun buildView(): TensorView<T, V> {
        // Determine the view type based on what slicing operations were specified
        return when {
            // Channel extraction optimization
            channelIndex != null -> {
                val mapper = NCHWViewHelper.createChannelExtraction(tensor.shape, channelIndex!!)
                // Create appropriate slice descriptors for channel extraction
                val descriptors = listOf(
                    SliceDescriptor.All, // batch
                    SliceDescriptor.Index(channelIndex!!), // channel
                    SliceDescriptor.All, // height
                    SliceDescriptor.All  // width
                )
                SlicedTensorView(tensor, mapper, descriptors)
            }
            
            // Batch slicing optimization
            batchSlice != null -> {
                val mapper = NCHWViewHelper.createBatchSlice(
                    tensor.shape, 
                    batchSlice!!.first, 
                    batchSlice!!.last + 1 // NCHWViewHelper expects exclusive end
                )
                val descriptors = listOf(
                    SliceDescriptor.Range(batchSlice!!.first, batchSlice!!.last + 1),
                    SliceDescriptor.All, // channel
                    SliceDescriptor.All, // height
                    SliceDescriptor.All  // width
                )
                SlicedTensorView(tensor, mapper, descriptors)
            }
            
            // Spatial slicing optimization
            heightSlice != null || widthSlice != null -> {
                val hRange = heightSlice ?: (0 until tensor.shape[2])
                val wRange = widthSlice ?: (0 until tensor.shape[3])
                val mapper = NCHWViewHelper.createSpatialSlice(
                    tensor.shape,
                    hRange.first, hRange.last + 1,
                    1, // height step
                    wRange.first, wRange.last + 1,
                    1  // width step
                )
                val descriptors = listOf(
                    SliceDescriptor.All, // batch
                    channelSlice?.let { SliceDescriptor.Range(it.first, it.last + 1) } ?: SliceDescriptor.All,
                    SliceDescriptor.Range(hRange.first, hRange.last + 1),
                    SliceDescriptor.Range(wRange.first, wRange.last + 1)
                )
                SlicedTensorView(tensor, mapper, descriptors)
            }
            
            // Default case - use general slice mapper
            else -> {
                val descriptors = listOf(
                    batchSlice?.let { SliceDescriptor.Range(it.first, it.last + 1) } ?: SliceDescriptor.All,
                    channelSlice?.let { SliceDescriptor.Range(it.first, it.last + 1) } ?: SliceDescriptor.All,
                    heightSlice?.let { SliceDescriptor.Range(it.first, it.last + 1) } ?: SliceDescriptor.All,
                    widthSlice?.let { SliceDescriptor.Range(it.first, it.last + 1) } ?: SliceDescriptor.All
                )
                val indexMapper = SliceIndexMapper(tensor.shape, descriptors)
                SlicedTensorView(tensor, indexMapper, descriptors)
            }
        }
    }
}

/**
 * Builder class for individual segment operations
 */
public class SegmentBuilder<T : DType, V>(
    private val tensor: Tensor<T, V>,
    private val dimensionIndex: Int
) {
    private var startIndex: Long = 0
    private var endIndex: Long = tensor.shape.dimensions[dimensionIndex].toLong() - 1

    /**
     * Selects all elements in this dimension (equivalent to ":" in numpy)
     */
    public fun all() {
        startIndex = 0
        endIndex = tensor.shape.dimensions[dimensionIndex].toLong() - 1
    }

    /**
     * Selects elements from the specified start index to the end
     */
    public fun from(start: Int) {
        val adjustedStart = if (start < 0) {
            tensor.shape.dimensions[dimensionIndex] + start
        } else {
            start
        }
        startIndex = adjustedStart.toLong()
        endIndex = tensor.shape.dimensions[dimensionIndex].toLong() - 1
    }

    /**
     * Selects elements from the beginning to the specified end index (exclusive)
     */
    public fun to(end: Int) {
        startIndex = 0
        val adjustedEnd = if (end < 0) {
            tensor.shape.dimensions[dimensionIndex] + end
        } else {
            end
        }
        endIndex = (adjustedEnd - 1).toLong()
    }

    /**
     * Selects elements in the specified range
     */
    public fun range(start: Int, end: Int) {
        val adjustedStart = if (start < 0) {
            tensor.shape.dimensions[dimensionIndex] + start
        } else {
            start
        }
        val adjustedEnd = if (end < 0) {
            tensor.shape.dimensions[dimensionIndex] + end
        } else {
            end
        }
        startIndex = adjustedStart.toLong()
        endIndex = (adjustedEnd - 1).toLong()
    }

    /**
     * Selects only the first element
     */
    public fun first() {
        startIndex = 0
        endIndex = 0
    }

    /**
     * Selects only the last element
     */
    public fun last() {
        val lastIdx = tensor.shape.dimensions[dimensionIndex].toLong() - 1
        startIndex = lastIdx
        endIndex = lastIdx
    }

    /**
     * Selects a single element at the specified index
     */
    public fun at(index: Int) {
        val adjustedIndex = if (index < 0) {
            tensor.shape.dimensions[dimensionIndex] + index
        } else {
            index
        }
        startIndex = adjustedIndex.toLong()
        endIndex = adjustedIndex.toLong()
    }

    /**
     * Builds the slice from the current segment configuration
     */
    internal fun build(): Slice<T, V> {
        // Ensure indices are within bounds
        val dimSize = tensor.shape.dimensions[dimensionIndex].toLong()
        val safeStartIndex = startIndex.coerceIn(0, dimSize - 1)
        val safeEndIndex = endIndex.coerceIn(0, dimSize - 1)
        
        return Slice(
            tensor = tensor,
            dimensionIndex = dimensionIndex,
            startIndex = safeStartIndex,
            endIndex = safeEndIndex
        )
    }
}

/**
 * Builder class for individual view segment operations
 */
public class ViewSegmentBuilder<T : DType, V>(
    private val tensor: Tensor<T, V>,
    private val dimensionIndex: Int
) {
    private var startIndex: Long = 0
    private var endIndex: Long = tensor.shape.dimensions[dimensionIndex].toLong() - 1
    private var isAllSlice = false
    private var isSingleIndex = false
    
    /**
     * Hint indicating whether this segment preserves memory contiguity.
     * Used for optimization decisions in view vs copy selection.
     */
    internal var isContiguousHint: Boolean = true
        private set
    
    /**
     * Hint indicating the complexity of this segment's access pattern.
     * Lower values indicate simpler, more efficient patterns.
     */
    internal var complexityHint: Int = 0
        private set

    /**
     * Selects all elements in this dimension (equivalent to ":" in numpy)
     */
    public fun all() {
        startIndex = 0
        endIndex = tensor.shape.dimensions[dimensionIndex].toLong() - 1
        isAllSlice = true
        isSingleIndex = false
        // All slices preserve contiguity and have minimal complexity
        isContiguousHint = true
        complexityHint = 0
    }

    /**
     * Selects elements from the specified start index to the end
     */
    public fun from(start: Int) {
        val adjustedStart = if (start < 0) {
            tensor.shape.dimensions[dimensionIndex] + start
        } else {
            start
        }
        startIndex = adjustedStart.toLong()
        endIndex = tensor.shape.dimensions[dimensionIndex].toLong() - 1
        isAllSlice = false
        isSingleIndex = false
        // From-slices maintain contiguity if starting from 0, otherwise moderate complexity
        isContiguousHint = (adjustedStart == 0)
        complexityHint = if (adjustedStart == 0) 1 else 3
    }

    /**
     * Selects elements from the beginning to the specified end index (exclusive)
     */
    public fun to(end: Int) {
        startIndex = 0
        val adjustedEnd = if (end < 0) {
            tensor.shape.dimensions[dimensionIndex] + end
        } else {
            end
        }
        endIndex = (adjustedEnd - 1).toLong()
        isAllSlice = false
        isSingleIndex = false
        // To-slices starting from 0 maintain good contiguity
        isContiguousHint = true
        complexityHint = 1
    }

    /**
     * Selects elements in the specified range
     */
    public fun range(start: Int, end: Int) {
        val adjustedStart = if (start < 0) {
            tensor.shape.dimensions[dimensionIndex] + start
        } else {
            start
        }
        val adjustedEnd = if (end < 0) {
            tensor.shape.dimensions[dimensionIndex] + end
        } else {
            end
        }
        startIndex = adjustedStart.toLong()
        endIndex = (adjustedEnd - 1).toLong()
        isAllSlice = false
        isSingleIndex = false
        // Range slices maintain contiguity if starting from 0, otherwise moderate complexity
        isContiguousHint = (adjustedStart == 0)
        complexityHint = if (adjustedStart == 0) 2 else 4
    }

    /**
     * Selects only the first element
     */
    public fun first() {
        startIndex = 0
        endIndex = 0
        isAllSlice = false
        isSingleIndex = true
        // Single index selections reduce dimensions, moderate complexity
        isContiguousHint = true
        complexityHint = 2
    }

    /**
     * Selects only the last element
     */
    public fun last() {
        val lastIdx = tensor.shape.dimensions[dimensionIndex].toLong() - 1
        startIndex = lastIdx
        endIndex = lastIdx
        isAllSlice = false
        isSingleIndex = true
        // Single index selections reduce dimensions, moderate complexity
        isContiguousHint = false
        complexityHint = 3
    }

    /**
     * Selects a single element at the specified index
     */
    public fun at(index: Int) {
        val adjustedIndex = if (index < 0) {
            tensor.shape.dimensions[dimensionIndex] + index
        } else {
            index
        }
        startIndex = adjustedIndex.toLong()
        endIndex = adjustedIndex.toLong()
        isAllSlice = false
        isSingleIndex = true
        // Single index selections reduce dimensions, complexity varies by position
        isContiguousHint = (adjustedIndex == 0)
        complexityHint = if (adjustedIndex == 0) 2 else 3
    }

    /**
     * Builds the slice descriptor from the current segment configuration
     */
    internal fun buildDescriptor(): SliceDescriptor {
        // Ensure indices are within bounds
        val dimSize = tensor.shape.dimensions[dimensionIndex].toLong()
        val safeStartIndex = startIndex.coerceIn(0, dimSize - 1).toInt()
        val safeEndIndex = endIndex.coerceIn(0, dimSize - 1).toInt()
        
        return when {
            isAllSlice -> SliceDescriptor.All
            isSingleIndex -> SliceDescriptor.Index(safeStartIndex)
            else -> SliceDescriptor.Range(safeStartIndex, safeEndIndex + 1) // +1 because Range is exclusive end
        }
    }
}
