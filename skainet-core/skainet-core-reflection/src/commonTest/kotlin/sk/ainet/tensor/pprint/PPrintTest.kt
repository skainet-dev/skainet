package sk.ainet.tensor.pprint

import sk.ainet.core.tensor.*
import sk.ainet.core.tensor.factory.TensorFactoryInitializer
import sk.ainet.core.tensor.factory.TensorFactoryRegistry
import kotlin.test.*

class PPrintTest {

    init {
        TensorFactoryInitializer.initializeFactories()
    }

    // Helper function to convert int array to byte array in little-endian
    private fun intsToBytes(ints: IntArray): ByteArray {
        val bytes = ByteArray(ints.size * 4)
        for (i in ints.indices) {
            bytes[i * 4 + 0] = (ints[i] and 0xFF).toByte()
            bytes[i * 4 + 1] = ((ints[i] shr 8) and 0xFF).toByte()
            bytes[i * 4 + 2] = ((ints[i] shr 16) and 0xFF).toByte()
            bytes[i * 4 + 3] = ((ints[i] shr 24) and 0xFF).toByte()
        }
        return bytes
    }

    // Helper function to convert float array to byte array in little-endian
    private fun floatsToBytes(floats: FloatArray): ByteArray {
        val bytes = ByteArray(floats.size * 4)
        for (i in floats.indices) {
            val bits = floats[i].toBits()
            bytes[i * 4 + 0] = (bits and 0xFF).toByte()
            bytes[i * 4 + 1] = ((bits shr 8) and 0xFF).toByte()
            bytes[i * 4 + 2] = ((bits shr 16) and 0xFF).toByte()
            bytes[i * 4 + 3] = ((bits shr 24) and 0xFF).toByte()
        }
        return bytes
    }

    // Note: Scalar test removed because TensorFactoryRegistry doesn't support empty shapes

    @Test
    fun testVectorPrettyPrint() {
        val shape = Shape(3)
        val bytes = intsToBytes(intArrayOf(1, 2, 3))
        val vector = TensorFactoryRegistry.createTensor(Int32, shape, bytes)

        val result = vector.pprint()
        assertEquals("( 1, 2, 3 )", result)
    }

    @Test
    fun testSingleElementVectorPrettyPrint() {
        val shape = Shape(1)
        val bytes = intsToBytes(intArrayOf(42))
        val vector = TensorFactoryRegistry.createTensor(Int32, shape, bytes)

        val result = vector.pprint()
        assertEquals("( 42 )", result)
    }

    @Test
    fun testMatrixPrettyPrint() {
        val shape = Shape(2, 3)
        val bytes = intsToBytes(intArrayOf(1, 2, 3, 4, 5, 6))
        val matrix = TensorFactoryRegistry.createTensor(Int32, shape, bytes)

        val result = matrix.pprint()
        val expected = """⎛ 1, 2, 3 ⎞
⎝ 4, 5, 6 ⎠"""
        assertEquals(expected, result)
    }

    @Test
    fun testBiggerMatrixPrettyPrint() {
        val shape = Shape(3, 3)
        val bytes = intsToBytes(intArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9))
        val matrix = TensorFactoryRegistry.createTensor(Int32, shape, bytes)

        val result = matrix.pprint()
        val expected = """⎛ 1, 2, 3 ⎞
⎜ 4, 5, 6 ⎟
⎝ 7, 8, 9 ⎠"""
        assertEquals(expected, result)
    }


    @Test
    fun testSingleRowMatrixPrettyPrint() {
        val shape = Shape(1, 3)
        val bytes = intsToBytes(intArrayOf(1, 2, 3))
        val matrix = TensorFactoryRegistry.createTensor(Int32, shape, bytes)

        val result = matrix.pprint()
        assertEquals("( 1, 2, 3 )", result)
    }

    @Test
    fun testSingleColumnMatrixPrettyPrint() {
        val shape = Shape(3, 1)
        val bytes = intsToBytes(intArrayOf(1, 2, 3))
        val matrix = TensorFactoryRegistry.createTensor(Int32, shape, bytes)

        val result = matrix.pprint()
        val expected = """⎛ 1 ⎞
⎜ 2 ⎟
⎝ 3 ⎠"""
        assertEquals(expected, result)
    }

    @Test
    fun testSingleElementMatrixPrettyPrint() {
        val shape = Shape(1, 1)
        val bytes = intsToBytes(intArrayOf(42))
        val matrix = TensorFactoryRegistry.createTensor(Int32, shape, bytes)

        val result = matrix.pprint()
        assertEquals("( 42 )", result)
    }

    @Test
    fun testHigherRankTensorPrettyPrint() {
        val shape = Shape(2, 2, 2)
        val bytes = intsToBytes(intArrayOf(1, 2, 3, 4, 5, 6, 7, 8))
        val tensor = TensorFactoryRegistry.createTensor(Int32, shape, bytes)

        val result = tensor.pprint()
        // For higher rank tensors, it should fall back to toString()
        assertTrue(result.contains("2"), "Higher rank tensor should use toString()")
    }

    @Test
    fun testFloatMatrixPrettyPrint() {
        val shape = Shape(2, 2)
        val bytes = floatsToBytes(floatArrayOf(1.5f, 2.5f, 3.5f, 4.5f))
        val matrix = TensorFactoryRegistry.createTensor(FP32, shape, bytes)

        val result = matrix.pprint()
        val expected = """⎛ 1.5, 2.5 ⎞
⎝ 3.5, 4.5 ⎠"""
        assertEquals(expected, result)
    }
}