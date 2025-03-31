package sk.ai.net

import kotlin.test.Test
import kotlin.test.assertEquals

class ShapeTest {

    @Test
    fun `test scalar`() {
        val shape = Shape(0)
        assertEquals(shape, Shape(1, 2))
    }
}