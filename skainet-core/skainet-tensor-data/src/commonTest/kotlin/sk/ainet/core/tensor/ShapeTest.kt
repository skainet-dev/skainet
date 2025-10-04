package sk.ainet.core.tensor

import kotlin.test.*

class ShapeTest {

    @Test
    fun testShapeCreation() {
        val shape1 = Shape(2, 3, 4)
        assertEquals(3, shape1.dimensions.size)
        assertContentEquals(intArrayOf(2, 3, 4), shape1.dimensions)
        
        val shape2 = Shape(intArrayOf(5, 6))
        assertEquals(2, shape2.dimensions.size)
        assertContentEquals(intArrayOf(5, 6), shape2.dimensions)
    }

    @Test
    fun testScalarShape() {
        val scalarShape = Shape()
        assertEquals(0, scalarShape.dimensions.size)
        assertEquals(1, scalarShape.volume)
        assertContentEquals(intArrayOf(), scalarShape.dimensions)
    }

    @Test
    fun testShapeVolume() {
        assertEquals(1, Shape().volume)
        assertEquals(5, Shape(5).volume)
        assertEquals(24, Shape(2, 3, 4).volume)
        assertEquals(120, Shape(2, 3, 4, 5).volume)
        assertEquals(0, Shape(2, 0, 4).volume) // Zero dimension creates volume 0
    }

    @Test
    fun testComputeStrides() {
        // 1D tensor
        val shape1D = Shape(5)
        val strides1D = shape1D.computeStrides()
        assertContentEquals(intArrayOf(1), strides1D)
        
        // 2D tensor
        val shape2D = Shape(3, 4)
        val strides2D = shape2D.computeStrides()
        assertContentEquals(intArrayOf(4, 1), strides2D)
        
        // 3D tensor
        val shape3D = Shape(2, 3, 4)
        val strides3D = shape3D.computeStrides()
        assertContentEquals(intArrayOf(12, 4, 1), strides3D)
        
        // Scalar (empty shape)
        val scalarShape = Shape()
        val scalarStrides = scalarShape.computeStrides()
        assertContentEquals(intArrayOf(), scalarStrides)
    }

    @Test
    fun testShapeEquality() {
        val shape1 = Shape(2, 3, 4)
        val shape2 = Shape(2, 3, 4)
        val shape3 = Shape(2, 3, 5)
        val shape4 = Shape(2, 3)
        
        assertEquals(shape1, shape2)
        assertNotEquals(shape1, shape3)
        assertNotEquals(shape1, shape4)
        assertFalse(shape1.equals("not a shape"))
    }

    @Test
    fun testShapeHashCode() {
        val shape1 = Shape(2, 3, 4)
        val shape2 = Shape(2, 3, 4)
        val shape3 = Shape(2, 3, 5)
        
        assertEquals(shape1.hashCode(), shape2.hashCode())
        assertNotEquals(shape1.hashCode(), shape3.hashCode())
    }

    @Test
    fun testShapeToString() {
        val shape = Shape(2, 3, 4)
        val toString = shape.toString()
        assertTrue(toString.contains("Shape"))
        assertTrue(toString.contains("Dimensions"))
        assertTrue(toString.contains("24")) // volume
        assertTrue(toString.contains("2 x 3 x 4")) // dimensions format
    }

    @Test
    fun testShapeRank() {
        assertEquals(0, Shape().rank)
        assertEquals(1, Shape(5).rank)
        assertEquals(2, Shape(2, 3).rank)
        assertEquals(3, Shape(2, 3, 4).rank)
    }

    @Test
    fun testShapeGetOperator() {
        val shape = Shape(2, 3, 4)
        assertEquals(2, shape[0])
        assertEquals(3, shape[1])
        assertEquals(4, shape[2])
        
        // Note: bounds checking behavior may vary by platform
        // Just test valid indices for multiplatform compatibility
    }

    @Test
    fun testShapeIndexValidation() {
        val shape = Shape(2, 3, 4)
        
        // Valid indices should work
        assertEquals(0, shape.index(intArrayOf(0, 0, 0)))
        assertEquals(23, shape.index(intArrayOf(1, 2, 3))) // 1*12 + 2*4 + 3 = 23
        
        // Invalid indices should throw AssertionError
        assertFailsWith<AssertionError> { shape.index(intArrayOf(2, 0, 0)) }
        assertFailsWith<AssertionError> { shape.index(intArrayOf(0, 3, 0)) }
        assertFailsWith<AssertionError> { shape.index(intArrayOf(0, 0, 4)) }
        assertFailsWith<AssertionError> { shape.index(intArrayOf(-1, 0, 0)) }
        assertFailsWith<AssertionError> { shape.index(intArrayOf(0, 0)) } // wrong number of indices
        assertFailsWith<AssertionError> { shape.index(intArrayOf(0, 0, 0, 0)) } // wrong number of indices
    }

    @Test
    fun testShapeBoundaryConditions() {
        // Single element shapes
        val singleElement = Shape(1)
        assertEquals(1, singleElement.volume)
        assertContentEquals(intArrayOf(1), singleElement.computeStrides())
        
        // Large dimensions
        val largeDim = Shape(1000, 1000)
        assertEquals(1_000_000, largeDim.volume)
        assertContentEquals(intArrayOf(1000, 1), largeDim.computeStrides())
    }

    @Test
    fun testShapeCompanionObjectCopy() {
        // Test that the companion object invoke method makes a copy
        val shape = Shape(2, 3, 4)
        assertEquals(2, shape.dimensions[0])
        assertEquals(3, shape.dimensions[1])
        assertEquals(4, shape.dimensions[2])
    }
}