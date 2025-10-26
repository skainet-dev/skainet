package sk.ainet.lang.tensor.dsl

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.tensor.testFactory
import sk.ainet.lang.types.Int32
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

class ImageNormalizationTest {
    @Test
    fun normalizeIntBCHWToUnitRange() {
        // Build a small BCHW tensor with Int32 pixels
        val values = intArrayOf(
            0, 127, 255,
            64, 128, 192
        )
        val intImage = tensor<Int32, Int>(testFactory) {
            shape(1, 1, 2, 3) {
                var idx = 0
                init { _ ->
                    val v = values[idx]
                    idx += 1
                    v
                }
            }
        }

        assertNotNull(intImage)
        assertEquals(Shape(1, 1, 2, 3), intImage.shape)

        // Normalize to [0,1]
        val norm = intImage.normalizePixels()

        assertNotNull(norm)
        assertEquals(Shape(1, 1, 2, 3), norm.shape)
        // dtype should be FP32
        assertEquals(FP32::class, norm.dtype)

        fun f(x: Int) = x.toFloat() / 255f
        // Check a few values
        assertEquals(f(0), norm.data[0, 0, 0, 0])
        assertEquals(f(127), norm.data[0, 0, 0, 1])
        assertEquals(f(255), norm.data[0, 0, 0, 2])
        assertEquals(f(64), norm.data[0, 0, 1, 0])
        assertEquals(f(128), norm.data[0, 0, 1, 1])
        assertEquals(f(192), norm.data[0, 0, 1, 2])
    }
}
