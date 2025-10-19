package sk.ainet.lang.tensor.factory

import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import sk.ainet.lang.types.Int4
import sk.ainet.lang.types.Int8
import sk.ainet.lang.types.Ternary
import kotlin.collections.get
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.text.get

class DenseTensorsTest {

    @Test
    fun createdIntTensorHasProperShape() {
        with(DenseTensorDataFactory()) {
            // scalar
            val scalar = TensorData.scalar<Int8, Int>(1)
            assertEquals(scalar.shape, Shape(1))
            assertEquals(scalar.shape.volume, 1)
            assertEquals(scalar[0], 1)

            // vector
            val vector = TensorData.vector<Int8, Int>(arrayOf(1, 2, 3))
            assertEquals(vector.shape, Shape(3))
            assertEquals(vector[0], 1)
            assertEquals(vector[1], 2)
            assertEquals(vector[2], 3)

            // matrix
            val matrix = TensorData.matrix<Int8, Int>(arrayOf(1, 2, 3), arrayOf(4, 5, 6))
            assertEquals(matrix.shape, Shape(2, 3))
            assertEquals(matrix[0, 0], 1)
            assertEquals(matrix[0, 1], 2)
            assertEquals(matrix[0, 2], 3)
            assertEquals(matrix[1, 0], 4)
            assertEquals(matrix[1, 1], 5)
            assertEquals(matrix[1, 2], 6)
        }
    }

    @Test
    fun createdFloatTensorHasProperShape() {
        with(DenseTensorDataFactory()) {
            // scalar
            val scalar = TensorData.scalar<FP32, Float>(1.0f)
            assertEquals(scalar.shape, Shape(1))
            assertEquals(scalar.shape.volume, 1)
            assertEquals(scalar[0], 1.0f)

            // vector
            val vector = TensorData.vector<FP32, Float>(arrayOf(1.0f, 2.0f, 3.0f))
            assertEquals(vector.shape, Shape(3))
            assertEquals(vector[0], 1.0f)
            assertEquals(vector[1], 2.0f)
            assertEquals(vector[2], 3.0f)

            // matrix
            val matrix = TensorData.matrix<FP32, Float>(arrayOf(1.0f, 2.0f, 3.0f), arrayOf(4.0f, 5.0f, 6.0f))
            assertEquals(matrix.shape, Shape(2, 3))
            assertEquals(matrix[0, 0], 1.0f)
            assertEquals(matrix[0, 1], 2.0f)
            assertEquals(matrix[0, 2], 3.0f)
            assertEquals(matrix[1, 0], 4.0f)
            assertEquals(matrix[1, 1], 5.0f)
            assertEquals(matrix[1, 2], 6.0f)

        }
    }

    @Test
    fun fromByteArrayFP32Works() {
        with(DenseTensorDataFactory()) {
            // Test FP32 fromByteArray with 1.0f in IEEE 754 little-endian
            val floatBytes = byteArrayOf(0x00, 0x00, 0x80.toByte(), 0x3F.toByte())
            val fp32TensorData = fromByteArray<FP32, Float>(floatBytes, FP32)

            assertEquals(Shape(1), fp32TensorData.shape)
            assertEquals(1.0f, fp32TensorData[0])
        }
    }

    @Test
    fun fromByteArrayInt8Works() {
        with(DenseTensorDataFactory()) {
            // Test Int8 fromByteArray
            val int8Bytes = byteArrayOf(42, -10, 100)
            val int8TensorData = fromByteArray<Int8, Byte>(int8Bytes, Int8)

            assertEquals(Shape(3), int8TensorData.shape)
            assertEquals(42.toByte(), int8TensorData[0])
            assertEquals((-10).toByte(), int8TensorData[1])
            assertEquals(100.toByte(), int8TensorData[2])
        }
    }

    @Test
    fun fromByteArrayInt4Works() {
        with(DenseTensorDataFactory()) {
            // Test Int4 fromByteArray - 0x5A = 0101 1010 -> lower nibble 10 (0xA), upper nibble 5
            val int4Bytes = byteArrayOf(0x5A.toByte())
            val int4TensorData = fromByteArray<Int4, Byte>(int4Bytes, Int4)

            assertEquals(Shape(2), int4TensorData.shape)
            assertEquals(10.toByte(), int4TensorData[0]) // Lower nibble: 0xA = 10
            assertEquals(5.toByte(), int4TensorData[1])  // Upper nibble: 0x5 = 5
        }
    }

    @Test
    fun fromByteArrayTernaryWorks() {
        with(DenseTensorDataFactory()) {
            // Test Ternary fromByteArray - 0x1B = 00011011 -> bits read as: 11 10 01 00 -> 0, 1, 0, -1
            val ternaryBytes = byteArrayOf(0x1B.toByte())
            val ternaryTensorData = fromByteArray<Ternary, Byte>(ternaryBytes, Ternary)

            assertEquals(Shape(4), ternaryTensorData.shape)
            assertEquals(0.toByte(), ternaryTensorData[0])    // bits 11 -> 0 (fallback)
            assertEquals(1.toByte(), ternaryTensorData[1])    // bits 10 -> 1
            assertEquals(0.toByte(), ternaryTensorData[2])    // bits 01 -> 0
            assertEquals((-1).toByte(), ternaryTensorData[3]) // bits 00 -> -1
        }
    }

    @Test
    fun fromFloatArrayFP32Works() {
        with(DenseTensorDataFactory()) {
            // Test FP32 fromFloatArray
            val floatData = floatArrayOf(1.5f, 2.5f, 3.5f)
            val fp32TensorData = fromFloatArray<FP32, Float>(floatData, FP32)

            assertEquals(Shape(3), fp32TensorData.shape)
            assertEquals(1.5f, fp32TensorData[0])
            assertEquals(2.5f, fp32TensorData[1])
            assertEquals(3.5f, fp32TensorData[2])
        }
    }

    @Test
    fun fromFloatArrayFP16Works() {
        with(DenseTensorDataFactory()) {
            // Test FP16 fromFloatArray
            val floatData = floatArrayOf(0.5f, -1.0f, 10.0f)
            val fp16TensorData = fromFloatArray<FP16, Float>(floatData, FP16)

            assertEquals(Shape(3), fp16TensorData.shape)
            assertEquals(0.5f, fp16TensorData[0])
            assertEquals(-1.0f, fp16TensorData[1])
            assertEquals(10.0f, fp16TensorData[2])
        }
    }

    @Test
    fun fromIntArrayInt32Works() {
        with(DenseTensorDataFactory()) {
            // Test Int32 fromIntArray
            val intData = intArrayOf(42, -100, 1000)
            val int32TensorData = fromIntArray<Int32, Int>(intData, Int32)

            assertEquals(Shape(3), int32TensorData.shape)
            assertEquals(42, int32TensorData[0])
            assertEquals(-100, int32TensorData[1])
            assertEquals(1000, int32TensorData[2])
        }
    }

    @Test
    fun fromFloatArrayUnsupportedDTypeThrows() {
        with(DenseTensorDataFactory()) {
            val floatData = floatArrayOf(1.0f, 2.0f)
            assertFailsWith<IllegalArgumentException> {
                fromFloatArray<Int8, Byte>(floatData, Int8)
            }
        }
    }

    @Test
    fun fromIntArrayUnsupportedDTypeThrows() {
        with(DenseTensorDataFactory()) {
            val intData = intArrayOf(1, 2)
            assertFailsWith<IllegalArgumentException> {
                fromIntArray<FP32, Float>(intData, FP32)
            }
        }
    }
}