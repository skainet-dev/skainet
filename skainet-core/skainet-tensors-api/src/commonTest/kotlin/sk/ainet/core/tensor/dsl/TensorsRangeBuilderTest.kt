package sk.ainet.core.tensor.dsl

import sk.ainet.core.tensor.DType
import sk.ainet.core.tensor.Int32
import sk.ainet.core.tensor.Shape
import sk.ainet.core.tensor.Tensor
import sk.ainet.core.tensor.TensorData
import sk.ainet.core.tensor.TensorFactory
import sk.ainet.core.tensor.TensorOps
import sk.ainet.core.tensor.backend.MockTensorFP32
import sk.ainet.core.tensor.backend.MockTensorInt32
import kotlin.test.Test
import kotlin.test.assertEquals


class TensorsRangeBuilderTest {

    private fun createMockTensor1D(size: Int): Tensor<Int32, Int> {
        val shape = Shape(intArrayOf(size))
        val data = Array(size) { it }
        return MockTensorInt32(shape, data.toIntArray())
    }

    private fun createMockTensor2D(rows: Int, cols: Int): Tensor<Int32, Int> {
        val shape = Shape(intArrayOf(rows, cols))
        val data = Array(rows * cols) { it }
        return MockTensorInt32(shape, data.toIntArray())
    }

    private fun createMockTensor3D(dim1: Int, dim2: Int, dim3: Int): Tensor<Int32, Int> {
        val shape = Shape(intArrayOf(dim1, dim2, dim3))
        val data = Array(dim1 * dim2 * dim3) { it }
        return MockTensorInt32(shape, data.toIntArray())
    }

    @Test
    fun testAllSegment() {
        val tensor = createMockTensor1D(10)
        val slices = slice(tensor) {
            segment {
                all()
            }
        }

        assertEquals(1, slices.size)
        val slice = slices[0]
        assertEquals(0, slice.startIndex)
        assertEquals(9, slice.endIndex) // all() should select from 0 to last index (9)
        assertEquals(0, slice.dimensionIndex)
    }

    @Test
    fun testFromSegment() {
        val tensor = createMockTensor1D(10)
        val slices = slice(tensor) {
            segment {
                from(3)
            }
        }

        assertEquals(1, slices.size)
        val slice = slices[0]
        assertEquals(3, slice.startIndex)
        assertEquals(9, slice.endIndex) // from 3 to end (9)
        assertEquals(0, slice.dimensionIndex)
    }

    @Test
    fun testFromWithNegativeIndex() {
        val tensor = createMockTensor1D(10)
        val slices = slice(tensor) {
            segment {
                from(-3) // should be 10 - 3 = 7
            }
        }

        assertEquals(1, slices.size)
        val slice = slices[0]
        assertEquals(7, slice.startIndex)
        assertEquals(9, slice.endIndex)
    }

    @Test
    fun testToSegment() {
        val tensor = createMockTensor1D(10)
        val slices = slice(tensor) {
            segment {
                to(5) // from 0 to 4 (exclusive)
            }
        }

        assertEquals(1, slices.size)
        val slice = slices[0]
        assertEquals(0, slice.startIndex)
        assertEquals(4, slice.endIndex)
    }

    @Test
    fun testToWithNegativeIndex() {
        val tensor = createMockTensor1D(10)
        val slices = slice(tensor) {
            segment {
                to(-2) // should be 10 - 2 = 8, so 0 to 7
            }
        }

        assertEquals(1, slices.size)
        val slice = slices[0]
        assertEquals(0, slice.startIndex)
        assertEquals(7, slice.endIndex)
    }

    @Test
    fun testRangeSegment() {
        val tensor = createMockTensor1D(10)
        val slices = slice(tensor) {
            segment {
                range(2, 7) // from 2 to 6 (exclusive end)
            }
        }

        assertEquals(1, slices.size)
        val slice = slices[0]
        assertEquals(2, slice.startIndex)
        assertEquals(6, slice.endIndex)
    }

    @Test
    fun testRangeWithNegativeIndices() {
        val tensor = createMockTensor1D(10)
        val slices = slice(tensor) {
            segment {
                range(-5, -2) // 10-5=5 to 10-2=8, so 5 to 7
            }
        }

        assertEquals(1, slices.size)
        val slice = slices[0]
        assertEquals(5, slice.startIndex)
        assertEquals(7, slice.endIndex)
    }

    @Test
    fun testFirstSegment() {
        val tensor = createMockTensor1D(10)
        val slices = slice(tensor) {
            segment {
                first()
            }
        }

        assertEquals(1, slices.size)
        val slice = slices[0]
        assertEquals(0, slice.startIndex)
        assertEquals(0, slice.endIndex)
    }

    @Test
    fun testLastSegment() {
        val tensor = createMockTensor1D(10)
        val slices = slice(tensor) {
            segment {
                last()
            }
        }

        assertEquals(1, slices.size)
        val slice = slices[0]
        assertEquals(9, slice.startIndex)
        assertEquals(9, slice.endIndex)
    }

    @Test
    fun testAtSegment() {
        val tensor = createMockTensor1D(10)
        val slices = slice(tensor) {
            segment {
                at(5)
            }
        }

        assertEquals(1, slices.size)
        val slice = slices[0]
        assertEquals(5, slice.startIndex)
        assertEquals(5, slice.endIndex)
    }

    @Test
    fun testAtWithNegativeIndex() {
        val tensor = createMockTensor1D(10)
        val slices = slice(tensor) {
            segment {
                at(-1) // should be 10 - 1 = 9
            }
        }

        assertEquals(1, slices.size)
        val slice = slices[0]
        assertEquals(9, slice.startIndex)
        assertEquals(9, slice.endIndex)
    }

    @Test
    fun testMultipleSegments2D() {
        val tensor = createMockTensor2D(5, 8)
        val slices = slice(tensor) {
            segment {
                all() // first dimension: all
            }
            segment {
                from(2) // second dimension: from 2 to end
            }
        }

        assertEquals(2, slices.size)

        val firstSlice = slices[0]
        assertEquals(0, firstSlice.dimensionIndex)
        assertEquals(0, firstSlice.startIndex)
        assertEquals(4, firstSlice.endIndex)

        val secondSlice = slices[1]
        assertEquals(1, secondSlice.dimensionIndex)
        assertEquals(2, secondSlice.startIndex)
        assertEquals(7, secondSlice.endIndex)
    }

    @Test
    fun testExampleFromIssueDescription() {
        val tensor = createMockTensor2D(5, 8)
        val slices = slice(tensor) {
            // from second to the last
            segment {
                all()
            }
            // all elements, equals to ":"
            // from 0 to the second last element
            segment {
                from(1)
            }
        }

        assertEquals(2, slices.size)

        val firstSlice = slices[0]
        assertEquals(0, firstSlice.dimensionIndex)
        assertEquals(0, firstSlice.startIndex)
        assertEquals(4, firstSlice.endIndex)

        val secondSlice = slices[1]
        assertEquals(1, secondSlice.dimensionIndex)
        assertEquals(1, secondSlice.startIndex)
        assertEquals(7, secondSlice.endIndex)
    }

    @Test
    fun testBoundsChecking() {
        val tensor = createMockTensor1D(5)
        val slices = slice(tensor) {
            segment {
                range(10, 20) // out of bounds, should be clamped
            }
        }

        assertEquals(1, slices.size)
        val slice = slices[0]
        // Should be clamped to valid range [0, 4]
        assertEquals(4, slice.startIndex) // clamped from 10
        assertEquals(4, slice.endIndex) // clamped from 19
    }

    @Test
    fun testNegativeStartIndexClampedToZero() {
        val tensor = createMockTensor1D(5)
        val slices = slice(tensor) {
            segment {
                range(-10, 3) // start should be clamped to 0
            }
        }

        assertEquals(1, slices.size)
        val slice = slices[0]
        assertEquals(0, slice.startIndex) // clamped from -10
        assertEquals(2, slice.endIndex) // 3-1 = 2
    }

    @Test
    fun testEmptyTensor() {
        val tensor = createMockTensor1D(1)
        val slices = slice(tensor) {
            segment {
                all()
            }
        }

        assertEquals(1, slices.size)
        val slice = slices[0]
        assertEquals(0, slice.startIndex)
        assertEquals(0, slice.endIndex)
    }

    @Test
    fun test3DTensor() {
        val tensor = createMockTensor3D(3, 4, 5)
        val slices = slice(tensor) {
            segment {
                at(1)
            }
            segment {
                range(1, 3)
            }
            segment {
                from(-2)
            }
        }

        assertEquals(3, slices.size)

        val firstSlice = slices[0]
        assertEquals(0, firstSlice.dimensionIndex)
        assertEquals(1, firstSlice.startIndex)
        assertEquals(1, firstSlice.endIndex)

        val secondSlice = slices[1]
        assertEquals(1, secondSlice.dimensionIndex)
        assertEquals(1, secondSlice.startIndex)
        assertEquals(2, secondSlice.endIndex)

        val thirdSlice = slices[2]
        assertEquals(2, thirdSlice.dimensionIndex)
        assertEquals(3, thirdSlice.startIndex) // 5 - 2 = 3
        assertEquals(4, thirdSlice.endIndex)
    }

    @Test
    fun testSliceDataStructure() {
        val tensor = createMockTensor1D(10)
        val slices = slice(tensor) {
            segment {
                range(2, 7)
            }
        }

        val slice = slices[0]
        assertEquals(tensor, slice.tensor)
        assertEquals(0, slice.dimensionIndex)
        assertEquals(2, slice.startIndex)
        assertEquals(6, slice.endIndex)
    }
}