package sk.ainet.core.tensor

import sk.ainet.core.tensor.backend.CpuTensorFP32
import sk.ainet.core.tensor.backend.CpuTensorInt8
import sk.ainet.core.tensor.backend.CpuTensorInt32
import sk.ainet.core.tensor.backend.CpuBackend
import sk.ainet.core.tensor.backend.CpuBackendInt8
import sk.ainet.core.tensor.backend.CpuBackendInt32
import kotlin.test.*

class ReshapeTest {
    
    private val backendFP32 = CpuBackend()
    private val backendInt8 = CpuBackendInt8()
    private val backendInt32 = CpuBackendInt32()

    @Test
    fun testReshape1DTo2D() {
        // Test reshaping 1D tensor [12] to 2D tensor [3, 4]
        val original = CpuTensorFP32.fromArray(
            Shape(12),
            floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f)
        )
        
        val reshaped = with(backendFP32) { original.reshape(Shape(3, 4)) }
        
        assertEquals(Shape(3, 4), reshaped.shape)
        assertEquals(12, reshaped.shape.volume)
        assertEquals(2, reshaped.shape.rank)
        
        // Verify data integrity - should be in row-major order
        assertEquals(1f, reshaped[0, 0])
        assertEquals(2f, reshaped[0, 1])
        assertEquals(3f, reshaped[0, 2])
        assertEquals(4f, reshaped[0, 3])
        assertEquals(5f, reshaped[1, 0])
        assertEquals(6f, reshaped[1, 1])
        assertEquals(7f, reshaped[1, 2])
        assertEquals(8f, reshaped[1, 3])
        assertEquals(9f, reshaped[2, 0])
        assertEquals(10f, reshaped[2, 1])
        assertEquals(11f, reshaped[2, 2])
        assertEquals(12f, reshaped[2, 3])
    }

    @Test
    fun testReshape2DTo1D() {
        // Test reshaping 2D tensor [2, 6] to 1D tensor [12]
        val original = CpuTensorFP32.fromArray(
            Shape(2, 6),
            floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f)
        )
        
        val reshaped = with(backendFP32) { original.reshape(Shape(12)) }
        
        assertEquals(Shape(12), reshaped.shape)
        assertEquals(12, reshaped.shape.volume)
        assertEquals(1, reshaped.shape.rank)
        
        // Verify data integrity
        for (i in 0 until 12) {
            assertEquals((i + 1).toFloat(), reshaped[i])
        }
    }

    @Test
    fun testReshape2DTo3D() {
        // Test reshaping 2D tensor [4, 6] to 3D tensor [2, 3, 4]
        val data = FloatArray(24) { (it + 1).toFloat() }
        val original = CpuTensorFP32.fromArray(Shape(4, 6), data)
        
        val reshaped = with(backendFP32) { original.reshape(Shape(2, 3, 4)) }
        
        assertEquals(Shape(2, 3, 4), reshaped.shape)
        assertEquals(24, reshaped.shape.volume)
        assertEquals(3, reshaped.shape.rank)
        
        // Verify some key data points
        assertEquals(1f, reshaped[0, 0, 0])
        assertEquals(2f, reshaped[0, 0, 1])
        assertEquals(5f, reshaped[0, 1, 0])
        assertEquals(24f, reshaped[1, 2, 3])
    }

    @Test
    fun testReshape3DTo2D() {
        // Test reshaping 3D tensor [2, 2, 3] to 2D tensor [6, 2]
        val data = FloatArray(12) { (it + 1).toFloat() }
        val original = CpuTensorFP32.fromArray(Shape(2, 2, 3), data)
        
        val reshaped = with(backendFP32) { original.reshape(Shape(6, 2)) }
        
        assertEquals(Shape(6, 2), reshaped.shape)
        assertEquals(12, reshaped.shape.volume)
        assertEquals(2, reshaped.shape.rank)
        
        // Verify data integrity
        assertEquals(1f, reshaped[0, 0])
        assertEquals(2f, reshaped[0, 1])
        assertEquals(3f, reshaped[1, 0])
        assertEquals(12f, reshaped[5, 1])
    }

    @Test
    fun testReshapeToSameShape() {
        // Test reshaping to the same shape (should work)
        val original = CpuTensorFP32.fromArray(
            Shape(2, 3),
            floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        )
        
        val reshaped = with(backendFP32) { original.reshape(Shape(2, 3)) }
        
        assertEquals(original.shape, reshaped.shape)
        assertEquals(original.shape.volume, reshaped.shape.volume)
        
        // Verify data integrity
        for (i in 0 until 2) {
            for (j in 0 until 3) {
                assertEquals(original[i, j], reshaped[i, j])
            }
        }
    }

    @Test
    fun testReshapeSingleElement() {
        // Test reshaping single element tensor
        val original = CpuTensorFP32.fromArray(Shape(1), floatArrayOf(42f))
        
        // Reshape to different single-element shapes
        val reshaped1 = with(backendFP32) { original.reshape(Shape(1, 1)) }
        assertEquals(Shape(1, 1), reshaped1.shape)
        assertEquals(42f, reshaped1[0, 0])
        
        val reshaped2 = with(backendFP32) { original.reshape(Shape(1, 1, 1)) }
        assertEquals(Shape(1, 1, 1), reshaped2.shape)
        assertEquals(42f, reshaped2[0, 0, 0])
    }

    @Test
    fun testReshapeHighDimensional() {
        // Test reshaping to higher dimensional tensor
        val data = FloatArray(24) { (it + 1).toFloat() }
        val original = CpuTensorFP32.fromArray(Shape(24), data)
        
        val reshaped = with(backendFP32) { original.reshape(Shape(2, 3, 2, 2)) }
        
        assertEquals(Shape(2, 3, 2, 2), reshaped.shape)
        assertEquals(24, reshaped.shape.volume)
        assertEquals(4, reshaped.shape.rank)
        
        // Verify some data points
        assertEquals(1f, reshaped[0, 0, 0, 0])
        assertEquals(2f, reshaped[0, 0, 0, 1])
        assertEquals(24f, reshaped[1, 2, 1, 1])
    }

    @Test
    fun testReshapeInt8Tensor() {
        // Test reshape with Int8 tensor
        val original = CpuTensorInt8.fromArray(
            Shape(6),
            byteArrayOf(1, 2, 3, 4, 5, 6)
        )
        
        val reshaped = with(backendInt8) { original.reshape(Shape(2, 3)) }
        
        assertEquals(Shape(2, 3), reshaped.shape)
        assertEquals(6, reshaped.shape.volume)
        assertEquals(2, reshaped.shape.rank)
        
        // Verify data integrity
        assertEquals(1.toByte(), reshaped[0, 0])
        assertEquals(2.toByte(), reshaped[0, 1])
        assertEquals(3.toByte(), reshaped[0, 2])
        assertEquals(4.toByte(), reshaped[1, 0])
        assertEquals(5.toByte(), reshaped[1, 1])
        assertEquals(6.toByte(), reshaped[1, 2])
    }

    @Test
    fun testReshapeInt32Tensor() {
        // Test reshape with Int32 tensor
        val original = CpuTensorInt32.fromArray(
            Shape(4),
            intArrayOf(10, 20, 30, 40)
        )
        
        val reshaped = with(backendInt32) { original.reshape(Shape(2, 2)) }
        
        assertEquals(Shape(2, 2), reshaped.shape)
        assertEquals(4, reshaped.shape.volume)
        assertEquals(2, reshaped.shape.rank)
        
        // Verify data integrity
        assertEquals(10, reshaped[0, 0])
        assertEquals(20, reshaped[0, 1])
        assertEquals(30, reshaped[1, 0])
        assertEquals(40, reshaped[1, 1])
    }

    @Test
    fun testReshapeVolumeValidation() {
        // Test that reshape fails when volumes don't match
        val tensor = CpuTensorFP32.fromArray(
            Shape(2, 3),
            floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        )
        
        assertFailsWith<IllegalArgumentException> {
            with(backendFP32) { tensor.reshape(Shape(2, 4)) } // Volume = 8, original volume = 6
        }
        
        assertFailsWith<IllegalArgumentException> {
            with(backendFP32) { tensor.reshape(Shape(7)) } // Volume = 7, original volume = 6
        }
        
        assertFailsWith<IllegalArgumentException> {
            with(backendFP32) { tensor.reshape(Shape(1, 1, 5)) } // Volume = 5, original volume = 6
        }
    }

    @Test
    fun testReshapeDataIntegrityComplex() {
        // Comprehensive test for data integrity with complex reshaping
        val data = FloatArray(60) { it.toFloat() }
        val original = CpuTensorFP32.fromArray(Shape(3, 4, 5), data)
        
        // Reshape through multiple dimensions
        val step1 = with(backendFP32) { original.reshape(Shape(12, 5)) }
        val step2 = with(backendFP32) { step1.reshape(Shape(60)) }
        val step3 = with(backendFP32) { step2.reshape(Shape(6, 10)) }
        val step4 = with(backendFP32) { step3.reshape(Shape(2, 3, 10)) }
        
        // Verify final shape
        assertEquals(Shape(2, 3, 10), step4.shape)
        
        // Verify data integrity by comparing flattened versions
        val originalFlat = Array<Float>(60) { 0f }
        val reshapedFlat = Array<Float>(60) { 0f }
        
        original.copyTo(originalFlat)
        step4.copyTo(reshapedFlat)
        
        for (i in 0 until 60) {
            assertEquals(originalFlat[i], reshapedFlat[i], "Mismatch at index $i")
        }
    }

    @Test
    fun testReshapeErrorMessages() {
        // Test that error messages contain useful information
        val tensor = CpuTensorFP32.fromArray(
            Shape(2, 3),
            floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        )
        
        val exception = assertFailsWith<IllegalArgumentException> {
            with(backendFP32) { tensor.reshape(Shape(2, 4)) }
        }
        
        // Verify error message contains volume information
        assertTrue(exception.message!!.contains("6"))
        assertTrue(exception.message!!.contains("8"))
    }

    @Test
    fun testReshapePreservesOriginal() {
        // Test that reshape doesn't modify the original tensor
        val original = CpuTensorFP32.fromArray(
            Shape(2, 3),
            floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        )
        
        val originalShape = original.shape
        val originalValue = original[0, 1]
        
        val reshaped = with(backendFP32) { original.reshape(Shape(3, 2)) }
        
        // Original should be unchanged
        assertEquals(originalShape, original.shape)
        assertEquals(originalValue, original[0, 1])
        
        // Reshaped should be different
        assertEquals(Shape(3, 2), reshaped.shape)
        assertNotEquals(originalShape, reshaped.shape)
    }

    @Test
    fun testReshapeChaining() {
        // Test that multiple reshape operations can be chained
        val original = CpuTensorFP32.fromArray(
            Shape(8),
            floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)
        )
        
        val reshaped1 = with(backendFP32) { original.reshape(Shape(2, 4)) }
        val reshaped2 = with(backendFP32) { reshaped1.reshape(Shape(4, 2)) }
        val reshaped3 = with(backendFP32) { reshaped2.reshape(Shape(2, 2, 2)) }
        val reshaped4 = with(backendFP32) { reshaped3.reshape(Shape(8)) }
        
        // Final shape should be back to original
        assertEquals(original.shape, reshaped4.shape)
        
        // Data should be preserved
        for (i in 0 until 8) {
            assertEquals(original[i], reshaped4[i])
        }
    }

    @Test
    fun testReshapeZeroVolume() {
        // Test edge case with zero volume tensors
        val tensor = CpuTensorFP32.fromArray(Shape(0), floatArrayOf())
        
        // Should be able to reshape to other zero-volume shapes
        val reshaped = with(backendFP32) { tensor.reshape(Shape(0, 5)) }
        assertEquals(Shape(0, 5), reshaped.shape)
        assertEquals(0, reshaped.shape.volume)
    }

    // Tests for reshape with -1 dimension inference
    @Test
    fun testReshapeWithMinusOneInference() {
        // Test reshaping 1D tensor [12] to 2D tensor [3, -1] -> [3, 4]
        val original = CpuTensorFP32.fromArray(
            Shape(12),
            floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f)
        )
        
        val reshaped = with(backendFP32) { original.reshape(3, -1) }
        
        assertEquals(Shape(3, 4), reshaped.shape)
        assertEquals(12, reshaped.shape.volume)
        assertEquals(2, reshaped.shape.rank)
        
        // Verify data integrity
        assertEquals(1f, reshaped[0, 0])
        assertEquals(2f, reshaped[0, 1])
        assertEquals(3f, reshaped[0, 2])
        assertEquals(4f, reshaped[0, 3])
        assertEquals(5f, reshaped[1, 0])
        assertEquals(12f, reshaped[2, 3])
    }

    @Test
    fun testReshapeWithMinusOneAtStart() {
        // Test reshaping 1D tensor [12] to 2D tensor [-1, 4] -> [3, 4]
        val original = CpuTensorFP32.fromArray(
            Shape(12),
            floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f)
        )
        
        val reshaped = with(backendFP32) { original.reshape(-1, 4) }
        
        assertEquals(Shape(3, 4), reshaped.shape)
        assertEquals(12, reshaped.shape.volume)
        
        // Verify data integrity
        assertEquals(1f, reshaped[0, 0])
        assertEquals(4f, reshaped[0, 3])
        assertEquals(5f, reshaped[1, 0])
        assertEquals(12f, reshaped[2, 3])
    }

    @Test
    fun testReshapeWithMinusOneThreeDimensions() {
        // Test reshaping 1D tensor [24] to 3D tensor [2, -1, 3] -> [2, 4, 3]
        val data = FloatArray(24) { (it + 1).toFloat() }
        val original = CpuTensorFP32.fromArray(Shape(24), data)
        
        val reshaped = with(backendFP32) { original.reshape(2, -1, 3) }
        
        assertEquals(Shape(2, 4, 3), reshaped.shape)
        assertEquals(24, reshaped.shape.volume)
        assertEquals(3, reshaped.shape.rank)
        
        // Verify some data points
        assertEquals(1f, reshaped[0, 0, 0])
        assertEquals(2f, reshaped[0, 0, 1])
        assertEquals(3f, reshaped[0, 0, 2])
        assertEquals(4f, reshaped[0, 1, 0])
        assertEquals(24f, reshaped[1, 3, 2])
    }

    @Test
    fun testReshapeWithMinusOneMiddlePosition() {
        // Test reshaping 2D tensor [6, 4] to 3D tensor [2, -1, 2] -> [2, 6, 2]
        val data = FloatArray(24) { (it + 1).toFloat() }
        val original = CpuTensorFP32.fromArray(Shape(6, 4), data)
        
        val reshaped = with(backendFP32) { original.reshape(2, -1, 2) }
        
        assertEquals(Shape(2, 6, 2), reshaped.shape)
        assertEquals(24, reshaped.shape.volume)
        
        // Verify data integrity
        assertEquals(1f, reshaped[0, 0, 0])
        assertEquals(24f, reshaped[1, 5, 1])
    }

    @Test
    fun testReshapeWithMinusOneInt8Tensor() {
        // Test reshape with -1 for Int8 tensor
        val original = CpuTensorInt8.fromArray(
            Shape(12),
            byteArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        )
        
        val reshaped = with(backendInt8) { original.reshape(3, -1) }
        
        assertEquals(Shape(3, 4), reshaped.shape)
        assertEquals(12, reshaped.shape.volume)
        
        // Verify data integrity
        assertEquals(1.toByte(), reshaped[0, 0])
        assertEquals(4.toByte(), reshaped[0, 3])
        assertEquals(12.toByte(), reshaped[2, 3])
    }

    @Test
    fun testReshapeWithMinusOneInt32Tensor() {
        // Test reshape with -1 for Int32 tensor
        val original = CpuTensorInt32.fromArray(
            Shape(8),
            intArrayOf(10, 20, 30, 40, 50, 60, 70, 80)
        )
        
        val reshaped = with(backendInt32) { original.reshape(-1, 2) }
        
        assertEquals(Shape(4, 2), reshaped.shape)
        assertEquals(8, reshaped.shape.volume)
        
        // Verify data integrity
        assertEquals(10, reshaped[0, 0])
        assertEquals(20, reshaped[0, 1])
        assertEquals(80, reshaped[3, 1])
    }

    @Test
    fun testReshapeWithoutMinusOne() {
        // Test that regular reshape still works with vararg version
        val original = CpuTensorFP32.fromArray(
            Shape(12),
            floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f)
        )
        
        val reshaped = with(backendFP32) { original.reshape(3, 4) }
        
        assertEquals(Shape(3, 4), reshaped.shape)
        assertEquals(12, reshaped.shape.volume)
        
        // Verify data integrity
        assertEquals(1f, reshaped[0, 0])
        assertEquals(12f, reshaped[2, 3])
    }

    @Test
    fun testReshapeMultipleMinusOneError() {
        // Test that multiple -1 dimensions cause an error
        val tensor = CpuTensorFP32.fromArray(
            Shape(12),
            floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f)
        )
        
        assertFailsWith<IllegalArgumentException> {
            with(backendFP32) { tensor.reshape(-1, -1) }
        }
        
        assertFailsWith<IllegalArgumentException> {
            with(backendFP32) { tensor.reshape(2, -1, -1) }
        }
    }

    @Test
    fun testReshapeMinusOneWithZeroDimension() {
        // Test that zero dimension with -1 causes an error
        val tensor = CpuTensorFP32.fromArray(
            Shape(12),
            floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f)
        )
        
        assertFailsWith<IllegalArgumentException> {
            with(backendFP32) { tensor.reshape(-1, 0) }
        }
    }

    @Test
    fun testReshapeMinusOneWithNegativeDimension() {
        // Test that negative dimension (other than -1) causes an error
        val tensor = CpuTensorFP32.fromArray(
            Shape(12),
            floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f)
        )
        
        assertFailsWith<IllegalArgumentException> {
            with(backendFP32) { tensor.reshape(-1, -2) }
        }
    }

    @Test
    fun testReshapeMinusOneInvalidDivision() {
        // Test that when remaining elements don't divide evenly, it causes an error
        val tensor = CpuTensorFP32.fromArray(
            Shape(13),  // Prime number
            FloatArray(13) { (it + 1).toFloat() }
        )
        
        assertFailsWith<IllegalArgumentException> {
            with(backendFP32) { tensor.reshape(-1, 4) } // 13 is not divisible by 4
        }
    }

    @Test
    fun testReshapeMinusOneDataIntegrityComplex() {
        // Complex test for data integrity with -1 inference
        val data = FloatArray(60) { it.toFloat() }
        val original = CpuTensorFP32.fromArray(Shape(60), data)
        
        // Reshape through multiple -1 inferences
        val step1 = with(backendFP32) { original.reshape(12, -1) } // [12, 5]
        val step2 = with(backendFP32) { step1.reshape(-1, 10) }    // [6, 10]
        val step3 = with(backendFP32) { step2.reshape(2, -1, 5) } // [2, 6, 5]
        
        // Verify final shape
        assertEquals(Shape(2, 6, 5), step3.shape)
        
        // Verify data integrity by checking some specific values
        assertEquals(0f, step3[0, 0, 0])
        assertEquals(1f, step3[0, 0, 1])
        assertEquals(59f, step3[1, 5, 4])
        
        // Verify by flattening back
        val flattened = Array<Float>(60) { 0f }
        step3.copyTo(flattened)
        
        for (i in 0 until 60) {
            assertEquals(i.toFloat(), flattened[i], "Mismatch at index $i")
        }
    }

    @Test
    fun testReshapeMinusOneErrorMessages() {
        // Test that error messages are informative
        val tensor = CpuTensorFP32.fromArray(
            Shape(12),
            floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f)
        )
        
        val exception1 = assertFailsWith<IllegalArgumentException> {
            with(backendFP32) { tensor.reshape(-1, -1) }
        }
        assertTrue(exception1.message!!.contains("Only one dimension can be -1"))
        
        val exception2 = assertFailsWith<IllegalArgumentException> {
            with(backendFP32) { tensor.reshape(-1, 5) }
        }
        assertTrue(exception2.message!!.contains("do not divide evenly"))
    }
}