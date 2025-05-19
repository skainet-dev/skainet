package sk.ai.net.nn

import sk.ai.net.Shape
import sk.ai.net.impl.DoublesTensor
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class FlattenTest {
    @Test
    fun `flatten 2d tensor`() {
        val tensor = DoublesTensor(
            Shape(2, 3),
            doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        )
        val flatten = Flatten()
        val result = flatten.forward(tensor) as DoublesTensor
        assertEquals(Shape(6), result.shape)
        assertContentEquals(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), result.elements)
    }

    @Test
    fun `flatten 3d tensor`() {
        val tensor = DoublesTensor(
            Shape(2, 2, 2),
            doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        )
        val flatten = Flatten()
        val result = flatten.forward(tensor) as DoublesTensor
        assertEquals(Shape(8), result.shape)
        assertContentEquals(
            doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
            result.elements
        )
    }
}
