package sk.ainet.core.tensor

import kotlin.test.*

class BroadcastTensorDataTest {

    @Test
    fun testBroadcastTensorDataConstruction1DTo2D() {
        val sourceData = DenseTensorData<FP32, Double>(
            Shape(3),
            arrayOf(1.0, 2.0, 3.0)
        )
        val broadcastShape = Shape(2, 3)
        
        val broadcastData = BroadcastTensorData(sourceData, broadcastShape)
        
        assertEquals(broadcastShape, broadcastData.shape)
        assertEquals(0, broadcastData.offset)
        assertFalse(broadcastData.isContiguous)
        assertContentEquals(intArrayOf(0, 1), broadcastData.strides)
    }

    @Test
    fun testBroadcastTensorDataConstruction2DTo3D() {
        val sourceData = DenseTensorData<FP32, Double>(
            Shape(1, 3),
            arrayOf(1.0, 2.0, 3.0)
        )
        val broadcastShape = Shape(2, 2, 3)
        
        val broadcastData = BroadcastTensorData(sourceData, broadcastShape)
        
        assertEquals(broadcastShape, broadcastData.shape)
        assertEquals(0, broadcastData.offset)
        assertFalse(broadcastData.isContiguous)
    }

    @Test
    fun testBroadcastTensorDataInvalidBroadcast() {
        val sourceData = DenseTensorData<FP32, Double>(
            Shape(3),
            arrayOf(1.0, 2.0, 3.0)
        )
        val invalidBroadcastShape = Shape(2, 2) // Cannot broadcast (3,) to (2, 2)
        
        assertFailsWith<IllegalArgumentException> {
            BroadcastTensorData(sourceData, invalidBroadcastShape)
        }
    }

    @Test
    fun testBroadcastTensorDataGet1DTo2D() {
        val sourceData = DenseTensorData<FP32, Double>(
            Shape(3),
            arrayOf(1.0, 2.0, 3.0)
        )
        val broadcastData = BroadcastTensorData(sourceData, Shape(2, 3))
        
        // First row should repeat the source data
        assertEquals(1.0, broadcastData[0, 0])
        assertEquals(2.0, broadcastData[0, 1])
        assertEquals(3.0, broadcastData[0, 2])
        
        // Second row should repeat the same data
        assertEquals(1.0, broadcastData[1, 0])
        assertEquals(2.0, broadcastData[1, 1])
        assertEquals(3.0, broadcastData[1, 2])
    }

    @Test
    fun testBroadcastTensorDataGetScalarTo2D() {
        val sourceData = DenseTensorData<FP32, Double>(
            Shape(),
            arrayOf(5.0)
        )
        val broadcastData = BroadcastTensorData(sourceData, Shape(2, 3))
        
        // All positions should have the scalar value
        assertEquals(5.0, broadcastData[0, 0])
        assertEquals(5.0, broadcastData[0, 1])
        assertEquals(5.0, broadcastData[0, 2])
        assertEquals(5.0, broadcastData[1, 0])
        assertEquals(5.0, broadcastData[1, 1])
        assertEquals(5.0, broadcastData[1, 2])
    }

    @Test
    fun testBroadcastTensorDataGetInvalidIndices() {
        val sourceData = DenseTensorData<FP32, Double>(
            Shape(3),
            arrayOf(1.0, 2.0, 3.0)
        )
        val broadcastData = BroadcastTensorData(sourceData, Shape(2, 3))
        
        // Test invalid dimensions count
        assertFailsWith<IllegalArgumentException> { broadcastData[0] }
        assertFailsWith<IllegalArgumentException> { broadcastData[0, 0, 0] }
        
        // Note: Out-of-bounds index tests are omitted as they test implementation details
        // rather than documented API behavior and may have platform-dependent exception types
    }

    @Test
    fun testBroadcastTensorDataCopyTo() {
        val sourceData = DenseTensorData<FP32, Double>(
            Shape(3),
            arrayOf(1.0, 2.0, 3.0)
        )
        val broadcastData = BroadcastTensorData(sourceData, Shape(2, 3))
        
        // Test copyTo without offset
        val dest1 = Array<Double>(6) { 0.0 }
        broadcastData.copyTo(dest1)
        // Should be broadcast pattern: [1,2,3,1,2,3]
        assertContentEquals(arrayOf(1.0, 2.0, 3.0, 1.0, 2.0, 3.0), dest1)
        
        // Test copyTo with offset
        val dest2 = Array<Double>(8) { 0.0 }
        broadcastData.copyTo(dest2, 1)
        assertContentEquals(arrayOf(0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 0.0), dest2)
    }

    @Test
    fun testBroadcastTensorDataSlice() {
        val sourceData = DenseTensorData<FP32, Double>(
            Shape(4),
            arrayOf(1.0, 2.0, 3.0, 4.0)
        )
        val broadcastData = BroadcastTensorData(sourceData, Shape(3, 4))
        
        // Create slice [1:3, 1:3] on broadcast data
        val ranges = intArrayOf(1, 3, 1, 3)
        val slicedData = broadcastData.slice(ranges)
        
        // Verify slice properties
        assertEquals(Shape(2, 2), slicedData.shape)
        
        // Verify slice content
        assertEquals(2.0, slicedData[0, 0]) // Was [1,1] in broadcast
        assertEquals(3.0, slicedData[0, 1]) // Was [1,2] in broadcast
        assertEquals(2.0, slicedData[1, 0]) // Was [2,1] in broadcast
        assertEquals(3.0, slicedData[1, 1]) // Was [2,2] in broadcast
    }

    @Test
    fun testBroadcastTensorDataSliceInvalidRanges() {
        val sourceData = DenseTensorData<FP32, Double>(
            Shape(3),
            arrayOf(1.0, 2.0, 3.0)
        )
        val broadcastData = BroadcastTensorData(sourceData, Shape(2, 3))
        
        // Test wrong number of ranges
        assertFailsWith<IllegalArgumentException> {
            broadcastData.slice(intArrayOf(0, 1)) // Too few ranges
        }
        
        assertFailsWith<IllegalArgumentException> {
            broadcastData.slice(intArrayOf(0, 1, 0, 1, 0, 1)) // Too many ranges
        }
        
        // Test invalid range values
        assertFailsWith<IllegalArgumentException> {
            broadcastData.slice(intArrayOf(-1, 1, 0, 2)) // Negative start
        }
        
        assertFailsWith<IllegalArgumentException> {
            broadcastData.slice(intArrayOf(0, 3, 0, 2)) // End beyond dimension
        }
        
        assertFailsWith<IllegalArgumentException> {
            broadcastData.slice(intArrayOf(1, 1, 0, 2)) // Start equals end
        }
    }

    @Test
    fun testBroadcastTensorDataMaterialize() {
        val sourceData = DenseTensorData<FP32, Double>(
            Shape(2),
            arrayOf(1.0, 2.0)
        )
        val broadcastData = BroadcastTensorData(sourceData, Shape(3, 2))
        
        val materialized = broadcastData.materialize()
        
        // Verify materialized properties
        assertEquals(Shape(3, 2), materialized.shape)
        assertTrue(materialized.isContiguous)
        assertTrue(materialized is DenseTensorData)
        
        // Verify materialized content
        assertEquals(1.0, materialized[0, 0])
        assertEquals(2.0, materialized[0, 1])
        assertEquals(1.0, materialized[1, 0])
        assertEquals(2.0, materialized[1, 1])
        assertEquals(1.0, materialized[2, 0])
        assertEquals(2.0, materialized[2, 1])
    }

    @Test
    fun testBroadcastTensorDataComplexBroadcast() {
        // Test broadcasting (2, 1, 3) to (4, 2, 5, 3)
        val sourceData = DenseTensorData<FP32, Double>(
            Shape(2, 1, 3),
            arrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        )
        val broadcastData = BroadcastTensorData(sourceData, Shape(4, 2, 5, 3))
        
        assertEquals(Shape(4, 2, 5, 3), broadcastData.shape)
        
        // Test key positions to verify broadcasting logic
        assertEquals(1.0, broadcastData[0, 0, 0, 0])
        assertEquals(2.0, broadcastData[0, 0, 0, 1])
        assertEquals(3.0, broadcastData[0, 0, 0, 2])
        assertEquals(4.0, broadcastData[0, 1, 0, 0])
        assertEquals(5.0, broadcastData[0, 1, 0, 1])
        assertEquals(6.0, broadcastData[0, 1, 0, 2])
        
        // Different positions that should broadcast the same values
        assertEquals(1.0, broadcastData[3, 0, 4, 0])
        assertEquals(2.0, broadcastData[3, 0, 4, 1])
        assertEquals(3.0, broadcastData[3, 0, 4, 2])
        assertEquals(4.0, broadcastData[3, 1, 4, 0])
    }

    @Test
    fun testCanBroadcastMethod() {
        // Test various broadcasting scenarios
        
        // Scalar to anything
        assertTrue(canBroadcast(Shape(), Shape(2, 3)))
        assertTrue(canBroadcast(Shape(), Shape(1)))
        
        // Same shapes
        assertTrue(canBroadcast(Shape(2, 3), Shape(2, 3)))
        
        // Valid broadcasts
        assertTrue(canBroadcast(Shape(3), Shape(2, 3)))
        assertTrue(canBroadcast(Shape(1, 3), Shape(2, 3)))
        assertTrue(canBroadcast(Shape(2, 1), Shape(2, 3)))
        assertTrue(canBroadcast(Shape(1, 1), Shape(2, 3)))
        
        // Invalid broadcasts
        assertFalse(canBroadcast(Shape(3), Shape(2, 2)))
        assertFalse(canBroadcast(Shape(2, 3), Shape(3, 2)))
        assertFalse(canBroadcast(Shape(2, 3), Shape(2)))
    }

    companion object {
        // Helper method to access the static canBroadcast method
        fun canBroadcast(sourceShape: Shape, targetShape: Shape): Boolean {
            // We need to access the private companion object method
            // For testing purposes, we'll create a dummy instance to test the validation
            return try {
                val dummySource = DenseTensorData<FP32, Double>(sourceShape, Array(sourceShape.volume) { 0.0 })
                BroadcastTensorData(dummySource, targetShape)
                true
            } catch (e: IllegalArgumentException) {
                false
            }
        }
    }
}