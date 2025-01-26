package io.github.kotlin.fibonacci

import sk.ai.net.Shape
import kotlin.test.Test
import kotlin.test.assertEquals

class ShapeTest {

    @Test
    fun `test scalar`() {
        val shape = Shape(0)
        assertEquals(shape, Shape(1, 2))
    }
}