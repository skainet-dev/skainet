package sk.ainet.core.tensor.dsl

import sk.ainet.core.tensor.DType
import sk.ainet.core.tensor.Int32
import sk.ainet.core.tensor.Shape
import sk.ainet.core.tensor.Tensor
import sk.ainet.core.tensor.TensorData
import kotlin.test.Test
import kotlin.test.assertEquals

// Mock tensor implementation for testing
class MockTensor<T : DType, V>(
    override val shape: Shape,
    private val data: Array<V>
) : Tensor<T, V> {
    
    // TensorData implementation
    override val strides: IntArray = shape.computeStrides()
    override val offset: Int = 0
    override val isContiguous: Boolean = true
    
    override fun get(vararg indices: Int): V {
        val index = shape.index(indices)
        return data[index]
    }
    
    override fun copyTo(dest: Array<V>, destOffset: Int) {
        for (i in data.indices) {
            dest[destOffset + i] = data[i]
        }
    }
    
    override fun slice(ranges: IntArray): TensorData<T, V> {
        require(ranges.size == shape.rank * 2) {
            "Ranges array must have size ${shape.rank * 2} (start,end pairs), got ${ranges.size}"
        }
        
        // Parse ranges and validate
        val sliceRanges = mutableListOf<Pair<Int, Int>>()
        for (i in 0 until shape.rank) {
            val start = ranges[i * 2]
            val end = ranges[i * 2 + 1]
            val dimSize = shape.dimensions[i]
            
            require(start >= 0 && start < dimSize) {
                "Start index $start out of bounds for dimension $i (size $dimSize)"
            }
            require(end > start && end <= dimSize) {
                "End index $end must be > start ($start) and <= dimension size ($dimSize)"
            }
            
            sliceRanges.add(start to end)
        }
        
        // Calculate new shape
        val newDimensions = sliceRanges.map { (start, end) -> end - start }.toIntArray()
        val newShape = Shape(newDimensions)
        
        // Create new data array for sliced tensor
        val newData = Array<Any?>(newShape.volume) { null }
        var destIndex = 0
        
        // Copy sliced data based on tensor rank
        when (shape.rank) {
            1 -> {
                val (start0, end0) = sliceRanges[0]
                for (i in start0 until end0) {
                    newData[destIndex++] = data[i]
                }
            }
            2 -> {
                val (start0, end0) = sliceRanges[0]
                val (start1, end1) = sliceRanges[1]
                for (i in start0 until end0) {
                    for (j in start1 until end1) {
                        val srcIndex = i * shape.dimensions[1] + j
                        newData[destIndex++] = data[srcIndex]
                    }
                }
            }
            3 -> {
                val (start0, end0) = sliceRanges[0]
                val (start1, end1) = sliceRanges[1]
                val (start2, end2) = sliceRanges[2]
                for (i in start0 until end0) {
                    for (j in start1 until end1) {
                        for (k in start2 until end2) {
                            val srcIndex = i * shape.dimensions[1] * shape.dimensions[2] +
                                         j * shape.dimensions[2] + k
                            newData[destIndex++] = data[srcIndex]
                        }
                    }
                }
            }
            else -> throw UnsupportedOperationException("MockTensor slicing only supports up to 3D tensors")
        }
        
        return MockTensor(newShape, newData as Array<V>)
    }
    
    override fun materialize(): TensorData<T, V> = this
    
    private fun Shape.computeStrides(): IntArray {
        if (dimensions.isEmpty()) return intArrayOf()
        val strides = IntArray(dimensions.size)
        strides[dimensions.size - 1] = 1
        for (i in dimensions.size - 2 downTo 0) {
            strides[i] = strides[i + 1] * dimensions[i + 1]
        }
        return strides
    }

    // TensorOps implementations - minimal for testing slice functionality
    override fun matmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> = this
    override fun matmul4d(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> = this
    override fun scale(a: Tensor<T, V>, scalar: Double): Tensor<T, V> = this
    override fun dot(a: Tensor<T, V>, b: Tensor<T, V>): Double = 0.0
    
    // Tensor-Tensor operations
    override fun Tensor<T, V>.plus(other: Tensor<T, V>): Tensor<T, V> = this@MockTensor
    override fun Tensor<T, V>.minus(other: Tensor<T, V>): Tensor<T, V> = this@MockTensor
    override fun Tensor<T, V>.times(other: Tensor<T, V>): Tensor<T, V> = this@MockTensor
    override fun Tensor<T, V>.div(other: Tensor<T, V>): Tensor<T, V> = this@MockTensor
    
    // Tensor-Scalar operations
    override fun Tensor<T, V>.plus(scalar: Int): Tensor<T, V> = this@MockTensor
    override fun Tensor<T, V>.minus(scalar: Int): Tensor<T, V> = this@MockTensor
    override fun Tensor<T, V>.times(scalar: Int): Tensor<T, V> = this@MockTensor
    override fun Tensor<T, V>.div(scalar: Int): Tensor<T, V> = this@MockTensor
    
    override fun Tensor<T, V>.plus(scalar: Float): Tensor<T, V> = this@MockTensor
    override fun Tensor<T, V>.minus(scalar: Float): Tensor<T, V> = this@MockTensor
    override fun Tensor<T, V>.times(scalar: Float): Tensor<T, V> = this@MockTensor
    override fun Tensor<T, V>.div(scalar: Float): Tensor<T, V> = this@MockTensor
    
    override fun Tensor<T, V>.plus(scalar: Double): Tensor<T, V> = this@MockTensor
    override fun Tensor<T, V>.minus(scalar: Double): Tensor<T, V> = this@MockTensor
    override fun Tensor<T, V>.times(scalar: Double): Tensor<T, V> = this@MockTensor
    override fun Tensor<T, V>.div(scalar: Double): Tensor<T, V> = this@MockTensor
    
    // Scalar-Tensor operations
    override fun Double.plus(t: Tensor<T, V>): Tensor<T, V> = this@MockTensor
    override fun Double.minus(t: Tensor<T, V>): Tensor<T, V> = this@MockTensor
    override fun Double.times(t: Tensor<T, V>): Tensor<T, V> = this@MockTensor
    override fun Double.div(t: Tensor<T, V>): Tensor<T, V> = this@MockTensor
    
    // Activation functions
    override fun Tensor<T, V>.t(): Tensor<T, V> = this@MockTensor
    override fun Tensor<T, V>.relu(): Tensor<T, V> = this@MockTensor
    override fun Tensor<T, V>.softmax(dimension: Int): Tensor<T, V> = this@MockTensor
    override fun Tensor<T, V>.sigmoid(): Tensor<T, V> = this@MockTensor
    override fun Tensor<T, V>.tanh(): Tensor<T, V> = this@MockTensor
    override fun Tensor<T, V>.flatten(startDim: Int, endDim: Int): Tensor<T, V> = this@MockTensor
    override fun Tensor<T, V>.reshape(newShape: Shape): Tensor<T, V> = this@MockTensor
    override fun Tensor<T, V>.reshape(vararg dimensions: Int): Tensor<T, V> = this@MockTensor
}

class TensorsRangeBuilderTest {

    private fun createMockTensor1D(size: Int): MockTensor<Int32, Int> {
        val shape = Shape(intArrayOf(size))
        val data = Array(size) { it }
        return MockTensor(shape, data)
    }

    private fun createMockTensor2D(rows: Int, cols: Int): MockTensor<Int32, Int> {
        val shape = Shape(intArrayOf(rows, cols))
        val data = Array(rows * cols) { it }
        return MockTensor(shape, data)
    }

    private fun createMockTensor3D(dim1: Int, dim2: Int, dim3: Int): MockTensor<Int32, Int> {
        val shape = Shape(intArrayOf(dim1, dim2, dim3))
        val data = Array(dim1 * dim2 * dim3) { it }
        return MockTensor(shape, data)
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