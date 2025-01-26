package io.github.kotlin.fibonacci

import sk.ai.net.Shape
import sk.ai.net.impl.createTensor
import kotlin.test.Test
import kotlin.test.assertEquals

class TensorTest {

    @Test
    fun testCreateTensor() {
        val dimensions = intArrayOf(2, 2)
        val values = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        val tensor = createTensor(Shape(*dimensions), values = values)

        assertEquals(2, tensor.shape.dimensions[0])
        assertEquals(2, tensor.shape.dimensions[1])
        assertEquals(2, tensor.rank)
        assertEquals(4, tensor.size)
        assertEquals(1.0, tensor.get(0, 0))
        assertEquals(2.0, tensor.get(0, 1))
        assertEquals(3.0, tensor.get(1, 0))
        assertEquals(4.0, tensor.get(1, 1))
    }
}