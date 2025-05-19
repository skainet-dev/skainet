package sk.ai.net.nn

import sk.ai.net.Shape
import sk.ai.net.impl.DoublesTensor
import kotlin.test.Test
import kotlin.test.assertEquals

class FlattenTest {
    @Test
    fun flatten_basic() {
        val flatten = Flatten()
        val input = DoublesTensor(Shape(2,1,28,28), DoubleArray(2*1*28*28))
        val output = flatten.forward(input) as DoublesTensor
        assertEquals(Shape(2,784), output.shape)
    }

    @Test
    fun flatten_with_custom_start_dim() {
        val flatten = Flatten(startDim = 1)
        val input = DoublesTensor(Shape(2,3,4), DoubleArray(2*3*4))
        val output = flatten.forward(input) as DoublesTensor
        assertEquals(Shape(2,12), output.shape)
    }

    @Test
    fun flatten_single_sample() {
        val flatten = Flatten()
        val input = DoublesTensor(Shape(1,3,3), DoubleArray(1*3*3))
        val output = flatten.forward(input) as DoublesTensor
        assertEquals(Shape(1,9), output.shape)
    }

    @Test
    fun flatten_preserve_batch_dim() {
        val flatten = Flatten()
        val input = DoublesTensor(Shape(10,5,2,2), DoubleArray(10*5*2*2))
        val output = flatten.forward(input) as DoublesTensor
        assertEquals(Shape(10,20), output.shape)
    }

    @Test
    fun flatten_no_batch_dim() {
        val flatten = Flatten()
        val input = DoublesTensor(Shape(1,3,4), DoubleArray(3*4))
        val output = flatten.forward(input) as DoublesTensor
        assertEquals(Shape(1,12), output.shape)
    }
}
