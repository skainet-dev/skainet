package sk.ainet.core.tensor

import sk.ainet.core.tensor.backend.CpuBackend
import sk.ainet.core.tensor.backend.CpuBackendInt8
import sk.ainet.core.tensor.backend.CpuBackendInt32
import sk.ainet.core.tensor.backend.CpuTensorFP32
import sk.ainet.core.tensor.backend.CpuTensorInt8
import sk.ainet.core.tensor.backend.CpuTensorInt32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class BroadcastingTest {

    // Note: Utility functions are private, so we test broadcasting indirectly through operations

    @Test
    fun testFP32BroadcastingAddition() {
        val backend = CpuBackend()
        
        // Test [2, 3] + [3] -> [2, 3]
        val a = CpuTensorFP32.fromArray(Shape(2, 3), floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
        val b = CpuTensorFP32.fromArray(Shape(3), floatArrayOf(10f, 20f, 30f))
        
        val result = with(backend) { a.plus(b) }
        assertEquals(Shape(2, 3), result.shape)
        
        val expectedData = floatArrayOf(11f, 22f, 33f, 14f, 25f, 36f)
        for (i in expectedData.indices) {
            assertEquals(expectedData[i], (result as CpuTensorFP32).data[i], 0.001f)
        }
    }

    @Test
    fun testFP32BroadcastingSubtraction() {
        val backend = CpuBackend()
        
        // Test [2, 3] - [3] -> [2, 3]
        val a = CpuTensorFP32.fromArray(Shape(2, 3), floatArrayOf(10f, 20f, 30f, 40f, 50f, 60f))
        val b = CpuTensorFP32.fromArray(Shape(3), floatArrayOf(1f, 2f, 3f))
        
        val result = with(backend) { a.minus(b) }
        assertEquals(Shape(2, 3), result.shape)
        
        val expectedData = floatArrayOf(9f, 18f, 27f, 39f, 48f, 57f)
        for (i in expectedData.indices) {
            assertEquals(expectedData[i], (result as CpuTensorFP32).data[i], 0.001f)
        }
    }

    @Test
    fun testFP32BroadcastingMultiplication() {
        val backend = CpuBackend()
        
        // Test [2, 3] * [3] -> [2, 3]
        val a = CpuTensorFP32.fromArray(Shape(2, 3), floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
        val b = CpuTensorFP32.fromArray(Shape(3), floatArrayOf(2f, 3f, 4f))
        
        val result = with(backend) { a.times(b) }
        assertEquals(Shape(2, 3), result.shape)
        
        val expectedData = floatArrayOf(2f, 6f, 12f, 8f, 15f, 24f)
        for (i in expectedData.indices) {
            assertEquals(expectedData[i], (result as CpuTensorFP32).data[i], 0.001f)
        }
    }

    @Test
    fun testFP32BroadcastingDivision() {
        val backend = CpuBackend()
        
        // Test [2, 3] / [3] -> [2, 3]
        val a = CpuTensorFP32.fromArray(Shape(2, 3), floatArrayOf(12f, 15f, 20f, 24f, 30f, 36f))
        val b = CpuTensorFP32.fromArray(Shape(3), floatArrayOf(3f, 5f, 4f))
        
        val result = with(backend) { a.div(b) }
        assertEquals(Shape(2, 3), result.shape)
        
        val expectedData = floatArrayOf(4f, 3f, 5f, 8f, 6f, 9f)
        for (i in expectedData.indices) {
            assertEquals(expectedData[i], (result as CpuTensorFP32).data[i], 0.001f)
        }
    }

    @Test
    fun testInt8BroadcastingAddition() {
        val backend = CpuBackendInt8()
        
        // Test [2, 2] + [2] -> [2, 2]
        val a = CpuTensorInt8.fromArray(Shape(2, 2), byteArrayOf(1, 2, 3, 4))
        val b = CpuTensorInt8.fromArray(Shape(2), byteArrayOf(10, 20))
        
        val result = with(backend) { a.plus(b) }
        assertEquals(Shape(2, 2), result.shape)
        
        val expectedData = byteArrayOf(11, 22, 13, 24)
        for (i in expectedData.indices) {
            assertEquals(expectedData[i], (result as CpuTensorInt8).data[i])
        }
    }

    @Test
    fun testInt32BroadcastingMultiplication() {
        val backend = CpuBackendInt32()
        
        // Test [2, 2] * [2] -> [2, 2]
        val a = CpuTensorInt32.fromArray(Shape(2, 2), intArrayOf(1, 2, 3, 4))
        val b = CpuTensorInt32.fromArray(Shape(2), intArrayOf(5, 6))
        
        val result = with(backend) { a.times(b) }
        assertEquals(Shape(2, 2), result.shape)
        
        val expectedData = intArrayOf(5, 12, 15, 24)
        for (i in expectedData.indices) {
            assertEquals(expectedData[i], (result as CpuTensorInt32).data[i])
        }
    }

    @Test
    fun testBroadcastingWith1DimensionalTensor() {
        val backend = CpuBackend()
        
        // Test [3, 4] + [1] -> [3, 4]
        val a = CpuTensorFP32.fromArray(
            Shape(3, 4),
            floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f)
        )
        val b = CpuTensorFP32.fromArray(Shape(1), floatArrayOf(100f))
        
        val result = with(backend) { a + b }
        assertEquals(Shape(3, 4), result.shape)
        
        val expectedData = floatArrayOf(101f, 102f, 103f, 104f, 105f, 106f, 107f, 108f, 109f, 110f, 111f, 112f)
        for (i in expectedData.indices) {
            assertEquals(expectedData[i], (result as CpuTensorFP32).data[i], 0.001f)
        }
    }

    @Test
    fun testBroadcastingWithMatrixAndVector() {
        val backend = CpuBackend()
        
        // Test [3, 1] + [1, 4] -> [3, 4]
        val a = CpuTensorFP32.fromArray(Shape(3, 1), floatArrayOf(1f, 2f, 3f))
        val b = CpuTensorFP32.fromArray(Shape(1, 4), floatArrayOf(10f, 20f, 30f, 40f))
        
        val result = with(backend) { a.plus(b) }
        assertEquals(Shape(3, 4), result.shape)
        
        val expectedData = floatArrayOf(
            11f, 21f, 31f, 41f,  // 1 + [10,20,30,40]
            12f, 22f, 32f, 42f,  // 2 + [10,20,30,40]
            13f, 23f, 33f, 43f   // 3 + [10,20,30,40]
        )
        for (i in expectedData.indices) {
            assertEquals(expectedData[i], (result as CpuTensorFP32).data[i], 0.001f)
        }
    }

    @Test
    fun testBroadcastingFailsForIncompatibleShapes() {
        val backend = CpuBackend()
        
        // Test incompatible shapes [3, 4] + [3, 5] should fail
        val a = CpuTensorFP32.fromArray(Shape(3, 4), FloatArray(12) { it.toFloat() })
        val b = CpuTensorFP32.fromArray(Shape(3, 5), FloatArray(15) { it.toFloat() })
        
        assertFailsWith<IllegalArgumentException> {
            with(backend) { a.plus(b) }
        }
    }

    @Test
    fun testBroadcastingWith3DTensors() {
        val backend = CpuBackend()
        
        // Test [2, 3, 4] + [3, 4] -> [2, 3, 4]
        val a = CpuTensorFP32.fromArray(Shape(2, 3, 4), FloatArray(24) { (it + 1).toFloat() })
        val b = CpuTensorFP32.fromArray(Shape(3, 4), FloatArray(12) { (it * 10).toFloat() })
        
        val result = with(backend) { a.plus(b) }
        assertEquals(Shape(2, 3, 4), result.shape)
        
        // Verify a few key values
        val resultData = (result as CpuTensorFP32).data
        assertEquals(1f + 0f, resultData[0], 0.001f)    // a[0,0,0] + b[0,0]
        assertEquals(2f + 10f, resultData[1], 0.001f)   // a[0,0,1] + b[0,1]
        assertEquals(13f + 0f, resultData[12], 0.001f)  // a[1,0,0] + b[0,0]
    }

    @Test
    fun testBroadcastingPreservesCompatibility() {
        val backend = CpuBackend()
        
        // Test that identical shapes work (no actual broadcasting needed)
        val a = CpuTensorFP32.fromArray(Shape(2, 3), floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
        val b = CpuTensorFP32.fromArray(Shape(2, 3), floatArrayOf(6f, 5f, 4f, 3f, 2f, 1f))
        
        val result = with(backend) { a.plus(b) }
        assertEquals(Shape(2, 3), result.shape)
        
        val expectedData = floatArrayOf(7f, 7f, 7f, 7f, 7f, 7f)
        for (i in expectedData.indices) {
            assertEquals(expectedData[i], (result as CpuTensorFP32).data[i], 0.001f)
        }
    }
}