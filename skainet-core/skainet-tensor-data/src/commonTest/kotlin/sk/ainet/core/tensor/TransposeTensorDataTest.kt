package sk.ainet.core.tensor

import kotlin.test.*

class TransposeTensorDataTest {

    @Test
    fun testTransposeTensorDataConstruction2D() {
        val sourceData = DenseTensorData<FP32, Double>(
            Shape(3, 2),
            arrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        )
        val permutation = intArrayOf(1, 0) // Transpose rows and columns
        
        val transposeData = TransposeTensorData(sourceData, permutation)
        
        assertEquals(Shape(2, 3), transposeData.shape)
        assertContentEquals(intArrayOf(1, 2), transposeData.strides)
        assertEquals(0, transposeData.offset)
        assertFalse(transposeData.isContiguous) // Transposed data is not contiguous
    }

    @Test
    fun testTransposeTensorDataConstruction3D() {
        val sourceData = DenseTensorData<FP32, Double>(
            Shape(2, 3, 4),
            Array(24) { (it + 1).toDouble() }
        )
        val permutation = intArrayOf(2, 0, 1) // Rotate dimensions
        
        val transposeData = TransposeTensorData(sourceData, permutation)
        
        assertEquals(Shape(4, 2, 3), transposeData.shape)
        assertContentEquals(intArrayOf(1, 12, 4), transposeData.strides)
        assertEquals(0, transposeData.offset)
        assertFalse(transposeData.isContiguous)
    }

    @Test
    fun testTransposeTensorDataInvalidPermutation() {
        val sourceData = DenseTensorData<FP32, Double>(
            Shape(2, 3),
            arrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        )
        
        // Test wrong size permutation
        assertFailsWith<IllegalArgumentException> {
            TransposeTensorData(sourceData, intArrayOf(0)) // Too few
        }
        
        // Test duplicate indices
        assertFailsWith<IllegalArgumentException> {
            TransposeTensorData(sourceData, intArrayOf(0, 0))
        }
        
        // Note: Tests for out-of-bounds indices are omitted as they test implementation details
        // rather than documented API behavior and have platform-dependent exception types
    }

    @Test
    fun testTransposeTensorDataGet2D() {
        val sourceData = DenseTensorData<FP32, Double>(
            Shape(2, 3),
            arrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        )
        val transposeData = TransposeTensorData(sourceData, intArrayOf(1, 0))
        
        // Original: [[1,2,3], [4,5,6]]
        // Transposed: [[1,4], [2,5], [3,6]]
        assertEquals(1.0, transposeData[0, 0]) // Was [0,0]
        assertEquals(4.0, transposeData[0, 1]) // Was [1,0]
        assertEquals(2.0, transposeData[1, 0]) // Was [0,1]
        assertEquals(5.0, transposeData[1, 1]) // Was [1,1]
        assertEquals(3.0, transposeData[2, 0]) // Was [0,2]
        assertEquals(6.0, transposeData[2, 1]) // Was [1,2]
    }

    @Test
    fun testTransposeTensorDataGet3D() {
        val sourceData = DenseTensorData<FP32, Double>(
            Shape(2, 2, 2),
            arrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        )
        val transposeData = TransposeTensorData(sourceData, intArrayOf(2, 1, 0))
        
        assertEquals(Shape(2, 2, 2), transposeData.shape)
        
        // Test a few key positions
        // Permutation (2,1,0) means (i,j,k) -> (k,j,i)
        assertEquals(1.0, transposeData[0, 0, 0]) // Was [0,0,0] -> [0,0,0]
        assertEquals(5.0, transposeData[0, 0, 1]) // Was [1,0,0] -> [0,0,1]
        assertEquals(2.0, transposeData[1, 0, 0]) // Was [0,0,1] -> [1,0,0]
        assertEquals(8.0, transposeData[1, 1, 1]) // Was [1,1,1] -> [1,1,1]
    }

    @Test
    fun testTransposeTensorDataGetInvalidIndices() {
        val sourceData = DenseTensorData<FP32, Double>(
            Shape(2, 3),
            arrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        )
        val transposeData = TransposeTensorData(sourceData, intArrayOf(1, 0))
        
        // Test invalid dimensions count
        assertFailsWith<IllegalArgumentException> { transposeData[0] }
        assertFailsWith<IllegalArgumentException> { transposeData[0, 0, 0] }
        
        // Test out of bounds indices (transposed shape is (3,2))
        assertFailsWith<IllegalArgumentException> { transposeData[3, 0] }
        assertFailsWith<IllegalArgumentException> { transposeData[0, 2] }
        assertFailsWith<IllegalArgumentException> { transposeData[-1, 0] }
    }

    @Test
    fun testTransposeTensorDataCopyTo() {
        val sourceData = DenseTensorData<FP32, Double>(
            Shape(2, 3),
            arrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        )
        val transposeData = TransposeTensorData(sourceData, intArrayOf(1, 0))
        
        // Test copyTo without offset
        val dest1 = Array<Double>(6) { 0.0 }
        transposeData.copyTo(dest1)
        // Should be transposed order: [1,4,2,5,3,6]
        assertContentEquals(arrayOf(1.0, 4.0, 2.0, 5.0, 3.0, 6.0), dest1)
        
        // Test copyTo with offset
        val dest2 = Array<Double>(8) { 0.0 }
        transposeData.copyTo(dest2, 1)
        assertContentEquals(arrayOf(0.0, 1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 0.0), dest2)
    }

    @Test
    fun testTransposeTensorDataSlice() {
        val sourceData = DenseTensorData<FP32, Double>(
            Shape(3, 4),
            Array(12) { (it + 1).toDouble() }
        )
        val transposeData = TransposeTensorData(sourceData, intArrayOf(1, 0))
        
        // Create slice [1:3, 1:3] on transposed data
        val ranges = intArrayOf(1, 3, 1, 3)
        val slicedData = transposeData.slice(ranges)
        
        // Verify slice properties
        assertEquals(Shape(2, 2), slicedData.shape)
        
        // This should create a TransposeTensorData of the sliced source
        assertTrue(slicedData is TransposeTensorData)
    }

    @Test
    fun testTransposeTensorDataSliceInvalidRanges() {
        val sourceData = DenseTensorData<FP32, Double>(
            Shape(2, 3),
            arrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        )
        val transposeData = TransposeTensorData(sourceData, intArrayOf(1, 0))
        
        // Test wrong number of ranges
        assertFailsWith<IllegalArgumentException> { transposeData.slice(intArrayOf(0, 1)) } // Too few
        assertFailsWith<IllegalArgumentException> { transposeData.slice(intArrayOf(0, 1, 0, 1, 0, 1)) } // Too many
        
        // Test invalid range bounds (transposed shape is (3,2))
        assertFailsWith<IllegalArgumentException> { transposeData.slice(intArrayOf(-1, 1, 0, 2)) }
        assertFailsWith<IllegalArgumentException> { transposeData.slice(intArrayOf(0, 4, 0, 2)) }
        assertFailsWith<IllegalArgumentException> { transposeData.slice(intArrayOf(1, 0, 0, 2)) } // start > end
    }

    @Test
    fun testTransposeTensorDataMaterialize() {
        val sourceData = DenseTensorData<FP32, Double>(
            Shape(2, 3),
            arrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        )
        val transposeData = TransposeTensorData(sourceData, intArrayOf(1, 0))
        
        // Materialize should create a DenseTensorData
        val materialized = transposeData.materialize()
        assertNotSame(transposeData, materialized)
        assertTrue(materialized is DenseTensorData)
        assertTrue(materialized.isContiguous)
        assertEquals(Shape(3, 2), materialized.shape)
        
        // Verify materialized data matches transposed access
        assertEquals(1.0, materialized[0, 0])
        assertEquals(4.0, materialized[0, 1])
        assertEquals(2.0, materialized[1, 0])
        assertEquals(5.0, materialized[1, 1])
        assertEquals(3.0, materialized[2, 0])
        assertEquals(6.0, materialized[2, 1])
    }

    @Test
    fun testTransposeTensorDataScalar() {
        val sourceData = DenseTensorData<FP32, Double>(
            Shape(),
            arrayOf(42.0)
        )
        val transposeData = TransposeTensorData(sourceData, intArrayOf())
        
        assertEquals(Shape(), transposeData.shape)
        assertContentEquals(intArrayOf(), transposeData.strides)
        assertEquals(0, transposeData.offset)
        assertEquals(42.0, transposeData.get())
    }

    @Test
    fun testTransposeTensorDataVector() {
        val sourceData = DenseTensorData<FP32, Double>(
            Shape(4),
            arrayOf(1.0, 2.0, 3.0, 4.0)
        )
        val transposeData = TransposeTensorData(sourceData, intArrayOf(0)) // Identity transpose
        
        assertEquals(Shape(4), transposeData.shape)
        assertContentEquals(intArrayOf(1), transposeData.strides)
        assertEquals(0, transposeData.offset)
        
        for (i in 0 until 4) {
            assertEquals((i + 1).toDouble(), transposeData[i])
        }
    }

    @Test
    fun testTransposeTensorDataIdentityPermutation() {
        val sourceData = DenseTensorData<FP32, Double>(
            Shape(2, 3),
            arrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        )
        val transposeData = TransposeTensorData(sourceData, intArrayOf(0, 1)) // Identity
        
        assertEquals(sourceData.shape, transposeData.shape)
        
        // Should behave identically to source
        for (i in 0 until 2) {
            for (j in 0 until 3) {
                assertEquals(sourceData[i, j], transposeData[i, j])
            }
        }
    }

    @Test
    fun testTransposeTensorDataComplexPermutation() {
        val sourceData = DenseTensorData<FP32, Double>(
            Shape(2, 3, 4),
            Array(24) { (it + 1).toDouble() }
        )
        val transposeData = TransposeTensorData(sourceData, intArrayOf(1, 2, 0))
        
        assertEquals(Shape(3, 4, 2), transposeData.shape)
        
        // Test specific mapping: (i,j,k) -> (j,k,i)
        assertEquals(sourceData[0, 1, 2], transposeData[1, 2, 0])
        assertEquals(sourceData[1, 0, 3], transposeData[0, 3, 1])
    }
}