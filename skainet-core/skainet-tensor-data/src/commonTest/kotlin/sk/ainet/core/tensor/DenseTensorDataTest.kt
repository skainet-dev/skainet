package sk.ainet.core.tensor

import kotlin.test.*

class DenseTensorDataTest {

    @Test
    fun testDenseTensorDataConstruction() {
        val shape = Shape(2, 3)
        val data = arrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        val tensorData = DenseTensorData<FP32, Double>(shape, data)
        
        assertEquals(shape, tensorData.shape)
        assertContentEquals(intArrayOf(3, 1), tensorData.strides)
        assertEquals(0, tensorData.offset)
        assertTrue(tensorData.isContiguous)
    }

    @Test
    fun testDenseTensorDataGet() {
        val shape = Shape(2, 3)
        val data = arrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        val tensorData = DenseTensorData<FP32, Double>(shape, data)
        
        // Test valid indices
        assertEquals(1.0, tensorData[0, 0])
        assertEquals(2.0, tensorData[0, 1])
        assertEquals(3.0, tensorData[0, 2])
        assertEquals(4.0, tensorData[1, 0])
        assertEquals(5.0, tensorData[1, 1])
        assertEquals(6.0, tensorData[1, 2])
    }

    @Test
    fun testDenseTensorDataGetInvalidIndices() {
        val shape = Shape(2, 3)
        val data = arrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        val tensorData = DenseTensorData<FP32, Double>(shape, data)
        
        // Test invalid dimensions count
        assertFailsWith<IllegalArgumentException> { tensorData[0] }
        assertFailsWith<IllegalArgumentException> { tensorData[0, 0, 0] }
        
        // Test out of bounds indices
        assertFailsWith<IllegalArgumentException> { tensorData[2, 0] }
        assertFailsWith<IllegalArgumentException> { tensorData[0, 3] }
        assertFailsWith<IllegalArgumentException> { tensorData[-1, 0] }
    }

    @Test
    fun testDenseTensorDataCopyTo() {
        val shape = Shape(2, 3)
        val data = arrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        val tensorData = DenseTensorData<FP32, Double>(shape, data)
        
        // Test copyTo without offset
        val dest1 = Array<Double>(6) { 0.0 }
        tensorData.copyTo(dest1)
        assertContentEquals(arrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), dest1)
        
        // Test copyTo with offset
        val dest2 = Array<Double>(10) { 0.0 }
        tensorData.copyTo(dest2, 2)
        assertContentEquals(arrayOf(0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0), dest2)
    }

    @Test
    fun testDenseTensorDataSlice() {
        val shape = Shape(3, 4)
        val data = Array(12) { (it + 1).toDouble() }
        val tensorData = DenseTensorData<FP32, Double>(shape, data)
        
        // Create slice [1:3, 1:3] -> 2x2 slice
        val ranges = intArrayOf(1, 3, 1, 3)
        val slicedData = tensorData.slice(ranges)
        
        // Verify slice properties
        assertEquals(Shape(2, 2), slicedData.shape)
        
        // Verify slice data
        assertEquals(6.0, slicedData[0, 0]) // Element at (1,1) in original
        assertEquals(7.0, slicedData[0, 1]) // Element at (1,2) in original
        assertEquals(10.0, slicedData[1, 0]) // Element at (2,1) in original
        assertEquals(11.0, slicedData[1, 1]) // Element at (2,2) in original
    }

    @Test
    fun testDenseTensorDataSliceInvalidRanges() {
        val shape = Shape(2, 3)
        val data = arrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        val tensorData = DenseTensorData<FP32, Double>(shape, data)
        
        // Test wrong number of ranges
        assertFailsWith<IllegalArgumentException> { tensorData.slice(intArrayOf(0, 1)) } // Too few
        assertFailsWith<IllegalArgumentException> { tensorData.slice(intArrayOf(0, 1, 0, 1, 0, 1)) } // Too many
        
        // Test invalid range bounds
        assertFailsWith<IllegalArgumentException> { tensorData.slice(intArrayOf(-1, 1, 0, 2)) }
        assertFailsWith<IllegalArgumentException> { tensorData.slice(intArrayOf(0, 3, 0, 2)) }
        assertFailsWith<IllegalArgumentException> { tensorData.slice(intArrayOf(1, 0, 0, 2)) } // start > end
    }

    @Test
    fun testDenseTensorDataMaterialize() {
        val shape = Shape(2, 3)
        val data = arrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        val tensorData = DenseTensorData<FP32, Double>(shape, data)
        
        // Since DenseTensorData is already contiguous, materialize should return the same instance
        val materialized = tensorData.materialize()
        assertSame(tensorData, materialized)
    }

    @Test
    fun testDenseTensorDataScalar() {
        val scalarShape = Shape()
        val scalarData = arrayOf(42.0)
        val tensorData = DenseTensorData<FP32, Double>(scalarShape, scalarData)
        
        assertEquals(scalarShape, tensorData.shape)
        assertContentEquals(intArrayOf(), tensorData.strides)
        assertEquals(0, tensorData.offset)
        assertTrue(tensorData.isContiguous)
        assertEquals(42.0, tensorData.get())
    }

    @Test
    fun testDenseTensorDataVector() {
        val vectorShape = Shape(5)
        val vectorData = arrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val tensorData = DenseTensorData<FP32, Double>(vectorShape, vectorData)
        
        assertEquals(vectorShape, tensorData.shape)
        assertContentEquals(intArrayOf(1), tensorData.strides)
        assertEquals(0, tensorData.offset)
        assertTrue(tensorData.isContiguous)
        
        for (i in 0 until 5) {
            assertEquals((i + 1).toDouble(), tensorData[i])
        }
    }

    @Test
    fun testDenseTensorData3D() {
        val shape3D = Shape(2, 2, 2)
        val data3D = Array(8) { (it + 1).toDouble() }
        val tensorData = DenseTensorData<FP32, Double>(shape3D, data3D)
        
        assertEquals(shape3D, tensorData.shape)
        assertContentEquals(intArrayOf(4, 2, 1), tensorData.strides)
        assertEquals(0, tensorData.offset)
        assertTrue(tensorData.isContiguous)
        
        assertEquals(1.0, tensorData[0, 0, 0])
        assertEquals(8.0, tensorData[1, 1, 1])
    }

    @Test
    fun testComputeStrides() {
        // Test the internal computeStrides extension function
        assertContentEquals(intArrayOf(), Shape().computeStrides())
        assertContentEquals(intArrayOf(1), Shape(5).computeStrides())
        assertContentEquals(intArrayOf(3, 1), Shape(2, 3).computeStrides())
        assertContentEquals(intArrayOf(12, 4, 1), Shape(2, 3, 4).computeStrides())
    }
}