package sk.ainet.core.tensor.dsl

import sk.ainet.core.tensor.DType
import sk.ainet.core.tensor.Slice
import sk.ainet.core.tensor.Tensor
import kotlin.math.abs

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
