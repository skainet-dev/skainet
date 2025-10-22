package org.mikrograd.samples

import org.mikrograd.diff.MLP
import org.mikrograd.diff.Value
import kotlin.math.PI
import kotlin.math.sin

fun train(sine: MLP) {

    //val X = listOf(0.0, PI / 2, PI)
    val X = List(100) { index ->
        (index / (100 - 1).toFloat()) * (PI / 2)
    }

    val y = X.mapIndexed { index, value -> Value(sin(value)).also { it.label = "y$index" } }



    (1..100).forEach {
        // forward propagation
        val ypred: List<List<Value>> = X.mapIndexed { index, x ->
            sine.invoke(listOf(x))//.also { it[0].label = "y_pred$index" }
        }
        // calculate loss
        val loss: Value = y.zip(ypred) { ygt, yout -> (ygt - yout[0]).pow(2.0) }.reduce { acc, v -> acc + v }
        // reset gradients
        sine.parameters().forEach { param ->
            param.grad = 0.0
        }
        // calc gradients in backpropagation
        loss.backward()

        // update weights and biases with a learning rate
        sine.parameters().forEach { param ->
            param.data += -0.1 * param.grad
        }

        println(loss.data)
    }
}

fun MLP.nsin(d: Double) = invoke(listOf(d))[0].data


fun main() {
    val sine = MLP(1, listOf(16, 16, 1))

    println(sine.nsin(PI / 2))
    train(sine)
    println(sine.nsin(PI / 2))
}


