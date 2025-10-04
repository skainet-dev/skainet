package sk.ainet.core.tensor

import kotlin.test.*

class ViewTensorDataTest {

    @Test
    fun testViewTensorDataConstruction() {
        val data = arrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        val shape = Shape(2, 2) // View as 2x2
        val strides = intArrayOf(3, 1) // Non-contiguous strides (skipping elements)
        val offset = 1
        val parentShape = Shape(8)
        
        val viewData = ViewTensorData<FP32, Double>(data, shape, strides, offset, parentShape)
        
        assertEquals(shape, viewData.shape)
        assertContentEquals(strides, viewData.strides)
        assertEquals(offset, viewData.offset)
        // With strides [3,1] for shape [2,2], expected would be [2,1], so not contiguous
        assertFalse(viewData.isContiguous)
    }

    @Test
    fun testViewTensorDataGet() {
        val data = arrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        val shape = Shape(2, 2)
        val strides = intArrayOf(3, 1) // Skip every 3rd element for rows
        val offset = 0
        val parentShape = Shape(2, 3)
        
        val viewData = ViewTensorData<FP32, Double>(data, shape, strides, offset, parentShape)
        
        // Test valid indices
        assertEquals(1.0, viewData[0, 0]) // data[0]
        assertEquals(2.0, viewData[0, 1]) // data[1]
        assertEquals(4.0, viewData[1, 0]) // data[3]
        assertEquals(5.0, viewData[1, 1]) // data[4]
    }

    @Test
    fun testViewTensorDataGetInvalidIndices() {
        val data = arrayOf(1.0, 2.0, 3.0, 4.0)
        val shape = Shape(2, 2)
        val strides = intArrayOf(2, 1)
        val offset = 0
        val parentShape = Shape(2, 2)
        
        val viewData = ViewTensorData<FP32, Double>(data, shape, strides, offset, parentShape)
        
        // Test invalid dimensions count
        assertFailsWith<IllegalArgumentException> { viewData[0] }
        assertFailsWith<IllegalArgumentException> { viewData[0, 0, 0] }
        
        // Test out of bounds indices
        assertFailsWith<IllegalArgumentException> { viewData[2, 0] }
        assertFailsWith<IllegalArgumentException> { viewData[0, 2] }
        assertFailsWith<IllegalArgumentException> { viewData[-1, 0] }
    }

    @Test
    fun testViewTensorDataContiguousCheck() {
        val data = arrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        val parentShape = Shape(2, 3)
        
        // Contiguous view (same as original)
        val contiguousShape = Shape(2, 3)
        val contiguousStrides = intArrayOf(3, 1)
        val contiguousView = ViewTensorData<FP32, Double>(data, contiguousShape, contiguousStrides, 0, parentShape)
        assertTrue(contiguousView.isContiguous)
        
        // Non-contiguous view (different strides)
        val nonContiguousShape = Shape(2, 2)
        val nonContiguousStrides = intArrayOf(3, 2)
        val nonContiguousView = ViewTensorData<FP32, Double>(data, nonContiguousShape, nonContiguousStrides, 0, parentShape)
        assertFalse(nonContiguousView.isContiguous)
    }

    @Test
    fun testViewTensorDataCopyTo() {
        val data = arrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        val shape = Shape(2, 2)
        val strides = intArrayOf(3, 1)
        val offset = 0
        val parentShape = Shape(2, 3)
        
        val viewData = ViewTensorData<FP32, Double>(data, shape, strides, offset, parentShape)
        
        // Test copyTo without offset
        val dest1 = Array<Double>(4) { 0.0 }
        viewData.copyTo(dest1)
        assertContentEquals(arrayOf(1.0, 2.0, 4.0, 5.0), dest1)
        
        // Test copyTo with offset
        val dest2 = Array<Double>(6) { 0.0 }
        viewData.copyTo(dest2, 1)
        assertContentEquals(arrayOf(0.0, 1.0, 2.0, 4.0, 5.0, 0.0), dest2)
    }

    @Test
    fun testViewTensorDataSlice() {
        val data = arrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
        val shape = Shape(3, 3)
        val strides = intArrayOf(3, 1)
        val offset = 0
        val parentShape = Shape(3, 3)
        
        val viewData = ViewTensorData<FP32, Double>(data, shape, strides, offset, parentShape)
        
        // Create slice [1:3, 1:3] -> 2x2 slice
        val ranges = intArrayOf(1, 3, 1, 3)
        val slicedData = viewData.slice(ranges)
        
        // Verify slice properties
        assertEquals(Shape(2, 2), slicedData.shape)
        
        // Verify slice data
        assertEquals(5.0, slicedData[0, 0]) // Element at (1,1) in original
        assertEquals(6.0, slicedData[0, 1]) // Element at (1,2) in original
        assertEquals(8.0, slicedData[1, 0]) // Element at (2,1) in original
        assertEquals(9.0, slicedData[1, 1]) // Element at (2,2) in original
    }

    @Test
    fun testViewTensorDataSliceInvalidRanges() {
        val data = arrayOf(1.0, 2.0, 3.0, 4.0)
        val shape = Shape(2, 2)
        val strides = intArrayOf(2, 1)
        val offset = 0
        val parentShape = Shape(2, 2)
        
        val viewData = ViewTensorData<FP32, Double>(data, shape, strides, offset, parentShape)
        
        // Test wrong number of ranges
        assertFailsWith<IllegalArgumentException> { viewData.slice(intArrayOf(0, 1)) } // Too few
        assertFailsWith<IllegalArgumentException> { viewData.slice(intArrayOf(0, 1, 0, 1, 0, 1)) } // Too many
        
        // Test invalid range bounds
        assertFailsWith<IllegalArgumentException> { viewData.slice(intArrayOf(-1, 1, 0, 2)) }
        assertFailsWith<IllegalArgumentException> { viewData.slice(intArrayOf(0, 3, 0, 2)) }
        assertFailsWith<IllegalArgumentException> { viewData.slice(intArrayOf(1, 0, 0, 2)) } // start > end
    }

    @Test
    fun testViewTensorDataMaterialize() {
        val data = arrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        val shape = Shape(2, 2)
        val strides = intArrayOf(3, 1)
        val offset = 0
        val parentShape = Shape(2, 3)
        
        val viewData = ViewTensorData<FP32, Double>(data, shape, strides, offset, parentShape)
        
        // Materialize should create a DenseTensorData
        val materialized = viewData.materialize()
        assertNotSame(viewData, materialized)
        assertTrue(materialized is DenseTensorData)
        assertTrue(materialized.isContiguous)
        assertEquals(shape, materialized.shape)
        
        // Verify materialized data
        assertEquals(1.0, materialized[0, 0])
        assertEquals(2.0, materialized[0, 1])
        assertEquals(4.0, materialized[1, 0])
        assertEquals(5.0, materialized[1, 1])
    }

    @Test
    fun testViewTensorDataScalar() {
        val data = arrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val scalarShape = Shape()
        val scalarStrides = intArrayOf()
        val offset = 2
        val parentShape = Shape(5)
        
        val viewData = ViewTensorData<FP32, Double>(data, scalarShape, scalarStrides, offset, parentShape)
        
        assertEquals(scalarShape, viewData.shape)
        assertContentEquals(scalarStrides, viewData.strides)
        assertEquals(offset, viewData.offset)
        assertEquals(3.0, viewData.get()) // data[2]
    }

    @Test
    fun testViewTensorDataVector() {
        val data = arrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        val vectorShape = Shape(4)
        val vectorStrides = intArrayOf(2) // Every other element
        val offset = 0
        val parentShape = Shape(8)
        
        val viewData = ViewTensorData<FP32, Double>(data, vectorShape, vectorStrides, offset, parentShape)
        
        assertEquals(vectorShape, viewData.shape)
        assertContentEquals(vectorStrides, viewData.strides)
        assertEquals(offset, viewData.offset)
        
        assertEquals(1.0, viewData[0]) // data[0]
        assertEquals(3.0, viewData[1]) // data[2]
        assertEquals(5.0, viewData[2]) // data[4]
        assertEquals(7.0, viewData[3]) // data[6]
    }

    @Test
    fun testViewTensorDataWithOffset() {
        val data = arrayOf(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        val shape = Shape(2, 2)
        val strides = intArrayOf(3, 1)
        val offset = 1 // Start from data[1]
        val parentShape = Shape(3, 3)
        
        val viewData = ViewTensorData<FP32, Double>(data, shape, strides, offset, parentShape)
        
        assertEquals(1.0, viewData[0, 0]) // data[1]
        assertEquals(2.0, viewData[0, 1]) // data[2]
        assertEquals(4.0, viewData[1, 0]) // data[4]
        assertEquals(5.0, viewData[1, 1]) // data[5]
    }
}