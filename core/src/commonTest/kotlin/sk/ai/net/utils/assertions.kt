package sk.ai.net.utils

import sk.ai.net.core.TypedTensor
import kotlin.math.abs

fun <T> assertTensorsSimilar(expected: TypedTensor<T>, actual: TypedTensor<T>, message: String? = null, comparator:(T, T) -> Boolean) {
    if (expected.shape != actual.shape) {
        throw AssertionError("Shapes are different: expected ${expected.shape}, actual ${actual.shape}")
    }
    if (expected.dataDescriptor != actual.dataDescriptor) {
        throw AssertionError("Data descriptors are different: expected ${expected.dataDescriptor}, actual ${actual.dataDescriptor}")
    }

    val expectedRanges = expected.shape.dimensions
    val actualRanges = actual.shape.dimensions
    // write code comparing the contents of the tensors by iterating over the elements by ranges of dimensions
    for (i in expectedRanges.indices) {
        val expectedRange = expectedRanges[i]
        val actualRange = actualRanges[i]
        if (expectedRange != actualRange) {
            throw AssertionError("Ranges are different: expected $expectedRange, actual $actualRange")
        }
    }

    expected.allElements.forEachIndexed { index, expectedElement: T ->
        if (!comparator(expectedElement, actual.allElements[index])) {
            throw AssertionError("Elements are different at index $index: expected $expectedElement, actual ${actual.allElements[index]}")
        }
    }
 }
