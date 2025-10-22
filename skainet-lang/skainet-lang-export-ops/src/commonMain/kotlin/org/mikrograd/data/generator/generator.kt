package org.mikrograd.data.generator

import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random

fun makeMoons(
    nSamples: Any = 100,
    shuffle: Boolean = true,
    noise: Double? = null,
    randomState: Int? = null
): Pair<Array<DoubleArray>, IntArray> {
    val (nSamplesOut, nSamplesIn) = when (nSamples) {
        is Int -> Pair(nSamples / 2, nSamples - nSamples / 2)
        is Pair<*, *> -> {
            if (nSamples.first is Int && nSamples.second is Int) {
                Pair(nSamples.first as Int, nSamples.second as Int)
            } else {
                throw IllegalArgumentException("`n_samples` can be either an int or a two-element tuple.")
            }
        }
        else -> throw IllegalArgumentException("`n_samples` can be either an int or a two-element tuple.")
    }

    val generator = randomState?.let { Random(it) } ?: Random.Default

    val outerCircX = DoubleArray(nSamplesOut) { cos(it * Math.PI / nSamplesOut) }
    val outerCircY = DoubleArray(nSamplesOut) { sin(it * Math.PI / nSamplesOut) }
    val innerCircX = DoubleArray(nSamplesIn) { 1 - cos(it * Math.PI / nSamplesIn) }
    val innerCircY = DoubleArray(nSamplesIn) { 1 - sin(it * Math.PI / nSamplesIn) - 0.5 }

    val X = Array(nSamplesOut + nSamplesIn) { DoubleArray(2) }
    for (i in 0 until nSamplesOut) {
        X[i][0] = outerCircX[i]
        X[i][1] = outerCircY[i]
    }
    for (i in 0 until nSamplesIn) {
        X[nSamplesOut + i][0] = innerCircX[i]
        X[nSamplesOut + i][1] = innerCircY[i]
    }

    val y = IntArray(nSamplesOut + nSamplesIn)
    for (i in 0 until nSamplesOut) {
        y[i] = 0
    }
    for (i in 0 until nSamplesIn) {
        y[nSamplesOut + i] = 1
    }

    if (shuffle) {
        val indices = X.indices.toList().shuffled(generator)
        val XShuffled = Array(X.size) { DoubleArray(2) }
        val yShuffled = IntArray(y.size)
        for (i in indices.indices) {
            XShuffled[i] = X[indices[i]]
            yShuffled[i] = y[indices[i]]
        }
        X.indices.forEach { X[it] = XShuffled[it] }
        y.indices.forEach { y[it] = yShuffled[it] }
    }

    noise?.let {
        for (i in X.indices) {
            X[i][0] += generator.nextDouble(-noise, noise)
            X[i][1] += generator.nextDouble(-noise, noise)
        }
    }

    return Pair(X, y)
}

