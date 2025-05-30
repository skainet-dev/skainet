package sk.ai.net

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals

class ShapeTest {

    @Test
    fun `test scalar`() {
        val shape = Shape(0)
        assertNotEquals(shape, Shape(1, 2))
    }
}