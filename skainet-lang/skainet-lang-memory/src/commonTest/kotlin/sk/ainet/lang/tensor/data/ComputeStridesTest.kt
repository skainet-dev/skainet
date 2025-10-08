package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import kotlin.test.*

class ComputeStridesTest {

    @Test
    fun testEmptyShape() {
        // Test empty shape (0 dimensions)
        val shape = Shape()
        val strides = shape.computeStrides()
        assertTrue(strides.isEmpty(), "Empty shape should return empty strides array")
    }

    @Test
    fun testScalarShape() {
        // Test scalar (shape with empty dimensions array, but this is effectively handled by empty case)
        val shape = Shape(intArrayOf())
        val strides = shape.computeStrides()
        assertTrue(strides.isEmpty(), "Scalar should return empty strides array")
    }

    @Test
    fun test1DShape() {
        // Test 1D shapes (vectors)
        val shape1 = Shape(5)
        val strides1 = shape1.computeStrides()
        assertContentEquals(intArrayOf(1), strides1, "1D shape [5] should have strides [1]")

        val shape2 = Shape(1)
        val strides2 = shape2.computeStrides()
        assertContentEquals(intArrayOf(1), strides2, "1D shape [1] should have strides [1]")

        val shape3 = Shape(10)
        val strides3 = shape3.computeStrides()
        assertContentEquals(intArrayOf(1), strides3, "1D shape [10] should have strides [1]")
    }

    @Test
    fun test2DShape() {
        // Test 2D shapes (matrices)
        val shape1 = Shape(3, 4)
        val strides1 = shape1.computeStrides()
        assertContentEquals(intArrayOf(4, 1), strides1, "2D shape [3, 4] should have strides [4, 1]")

        val shape2 = Shape(2, 5)
        val strides2 = shape2.computeStrides()
        assertContentEquals(intArrayOf(5, 1), strides2, "2D shape [2, 5] should have strides [5, 1]")

        val shape3 = Shape(1, 1)
        val strides3 = shape3.computeStrides()
        assertContentEquals(intArrayOf(1, 1), strides3, "2D shape [1, 1] should have strides [1, 1]")
    }

    @Test
    fun test3DShape() {
        // Test 3D shapes
        val shape1 = Shape(2, 3, 4)
        val strides1 = shape1.computeStrides()
        assertContentEquals(intArrayOf(12, 4, 1), strides1, "3D shape [2, 3, 4] should have strides [12, 4, 1]")

        val shape2 = Shape(3, 2, 5)
        val strides2 = shape2.computeStrides()
        assertContentEquals(intArrayOf(10, 5, 1), strides2, "3D shape [3, 2, 5] should have strides [10, 5, 1]")

        val shape3 = Shape(1, 1, 1)
        val strides3 = shape3.computeStrides()
        assertContentEquals(intArrayOf(1, 1, 1), strides3, "3D shape [1, 1, 1] should have strides [1, 1, 1]")
    }

    @Test
    fun test4DShape() {
        // Test 4D shapes
        val shape1 = Shape(2, 3, 4, 5)
        val strides1 = shape1.computeStrides()
        assertContentEquals(intArrayOf(60, 20, 5, 1), strides1, "4D shape [2, 3, 4, 5] should have strides [60, 20, 5, 1]")

        val shape2 = Shape(1, 2, 3, 4)
        val strides2 = shape2.computeStrides()
        assertContentEquals(intArrayOf(24, 12, 4, 1), strides2, "4D shape [1, 2, 3, 4] should have strides [24, 12, 4, 1]")
    }

    @Test
    fun testHigherDimensionalShapes() {
        // Test 5D shape
        val shape5D = Shape(2, 3, 4, 5, 6)
        val strides5D = shape5D.computeStrides()
        assertContentEquals(intArrayOf(360, 120, 30, 6, 1), strides5D, 
            "5D shape [2, 3, 4, 5, 6] should have strides [360, 120, 30, 6, 1]")

        // Test 6D shape
        val shape6D = Shape(1, 2, 1, 3, 1, 4)
        val strides6D = shape6D.computeStrides()
        assertContentEquals(intArrayOf(24, 12, 12, 4, 4, 1), strides6D,
            "6D shape [1, 2, 1, 3, 1, 4] should have strides [24, 12, 12, 4, 4, 1]")
    }

    @Test
    fun testEdgeCasesWithDimensionOfSize1() {
        // Test shapes with dimension of size 1 in various positions
        val shape1 = Shape(1, 5)
        val strides1 = shape1.computeStrides()
        assertContentEquals(intArrayOf(5, 1), strides1, "Shape [1, 5] should have strides [5, 1]")

        val shape2 = Shape(5, 1)
        val strides2 = shape2.computeStrides()
        assertContentEquals(intArrayOf(1, 1), strides2, "Shape [5, 1] should have strides [1, 1]")

        val shape3 = Shape(3, 1, 4)
        val strides3 = shape3.computeStrides()
        assertContentEquals(intArrayOf(4, 4, 1), strides3, "Shape [3, 1, 4] should have strides [4, 4, 1]")

        val shape4 = Shape(1, 3, 1, 4, 1)
        val strides4 = shape4.computeStrides()
        assertContentEquals(intArrayOf(12, 4, 4, 1, 1), strides4, 
            "Shape [1, 3, 1, 4, 1] should have strides [12, 4, 4, 1, 1]")
    }

    @Test
    fun testRowMajorOrdering() {
        // Verify that the strides follow row-major ordering
        // For a shape [d0, d1, d2, ..., dn], the stride for dimension i should be
        // the product of all dimensions to the right: d(i+1) * d(i+2) * ... * dn
        
        val shape = Shape(2, 3, 4, 5)
        val strides = shape.computeStrides()
        val dimensions = shape.dimensions
        
        // Manually calculate expected strides
        val expectedStrides = IntArray(dimensions.size)
        expectedStrides[dimensions.size - 1] = 1
        for (i in dimensions.size - 2 downTo 0) {
            expectedStrides[i] = expectedStrides[i + 1] * dimensions[i + 1]
        }
        
        assertContentEquals(expectedStrides, strides, "Strides should follow row-major ordering")
    }

    @Test
    fun testStrideCalculationCorrectness() {
        // Test that strides can be used to correctly calculate linear indices
        val shape = Shape(2, 3, 4)
        val strides = shape.computeStrides()
        
        // For index [1, 2, 3] in shape [2, 3, 4], the linear index should be:
        // 1*12 + 2*4 + 3*1 = 12 + 8 + 3 = 23
        val linearIndex = 1 * strides[0] + 2 * strides[1] + 3 * strides[2]
        assertEquals(23, linearIndex, "Linear index calculation should be correct")
        
        // Test corner case [0, 0, 0]
        val linearIndex2 = 0 * strides[0] + 0 * strides[1] + 0 * strides[2]
        assertEquals(0, linearIndex2, "Linear index for [0, 0, 0] should be 0")
    }
}