package sk.ainet.nn.dsl.extensions

import sk.ainet.core.tensor.*
import sk.ainet.core.tensor.factory.fromBytes
import sk.ainet.core.tensor.factory.ByteArrayConverter
import sk.ainet.nn.dsl.WeightsScope
import sk.ainet.nn.dsl.BiasScope

/**
 * DSL Extensions for Enhanced Tensor Creation
 *
 * This file provides convenient extension functions for creating tensors from primitive arrays
 * in a backend-agnostic way within the neural network DSL. These extensions support the
 * cleaner syntax requested for weight and bias initialization.
 *
 * Supports:
 * - Float, Double, and Int data types
 * - Vector format (varargs)
 * - Matrix format (List of Lists and Array of Arrays)
 * - Backend independence through factory registry
 */

// ==============================================
// Float Array Extensions
// ==============================================

/**
 * Creates a tensor from Float varargs (vector format).
 * Usage: fromArray(0.1f, -0.1f, 0.2f, -0.2f)
 */
public fun WeightsScope<FP32, Float>.fromArray(vararg values: Float): Tensor<FP32, Float> {
    val floatArray = values
    val byteData = ByteArrayConverter.convertFloatArrayToBytes(floatArray)
    return fromBytes(FP32, shape, byteData) as Tensor<FP32, Float>
}

public fun BiasScope<FP32, Float>.fromArray(vararg values: Float): Tensor<FP32, Float> {
    val floatArray = values
    val byteData = ByteArrayConverter.convertFloatArrayToBytes(floatArray)
    return fromBytes(FP32, shape, byteData) as Tensor<FP32, Float>
}

/**
 * Creates a tensor from Float array (explicit array).
 * Usage: fromFloatArray(floatArrayOf(0.1f, -0.1f, 0.2f))
 */
public fun WeightsScope<FP32, Float>.fromFloatArray(values: FloatArray): Tensor<FP32, Float> {
    val byteData = ByteArrayConverter.convertFloatArrayToBytes(values)
    return fromBytes(FP32, shape, byteData) as Tensor<FP32, Float>
}

public fun BiasScope<FP32, Float>.fromFloatArray(values: FloatArray): Tensor<FP32, Float> {
    val byteData = ByteArrayConverter.convertFloatArrayToBytes(values)
    return fromBytes(FP32, shape, byteData) as Tensor<FP32, Float>
}

/**
 * Creates a tensor from matrix format (List of Lists).
 * Usage: fromFloatMatrix(
 *   listOf(0.1f, -0.1f, 0.2f),
 *   listOf(-0.1f, 0.2f, -0.2f)
 * )
 */
public fun WeightsScope<FP32, Float>.fromFloatMatrix(vararg rows: List<Float>): Tensor<FP32, Float> {
    val flatArray = rows.flatMap { it }.toFloatArray()
    val byteData = ByteArrayConverter.convertFloatArrayToBytes(flatArray)
    return fromBytes(FP32, shape, byteData) as Tensor<FP32, Float>
}

public fun BiasScope<FP32, Float>.fromFloatMatrix(vararg rows: List<Float>): Tensor<FP32, Float> {
    val flatArray = rows.flatMap { it }.toFloatArray()
    val byteData = ByteArrayConverter.convertFloatArrayToBytes(flatArray)
    return fromBytes(FP32, shape, byteData) as Tensor<FP32, Float>
}

/**
 * Creates a tensor from matrix format (Array of Arrays).
 * Usage: fromFloatArrayMatrix(
 *   floatArrayOf(0.1f, -0.1f, 0.2f),
 *   floatArrayOf(-0.1f, 0.2f, -0.2f)
 * )
 */
public fun WeightsScope<FP32, Float>.fromFloatArrayMatrix(vararg rows: FloatArray): Tensor<FP32, Float> {
    val flatArray = rows.flatMap { it.asIterable() }.toFloatArray()
    val byteData = ByteArrayConverter.convertFloatArrayToBytes(flatArray)
    return fromBytes(FP32, shape, byteData) as Tensor<FP32, Float>
}

public fun BiasScope<FP32, Float>.fromFloatArrayMatrix(vararg rows: FloatArray): Tensor<FP32, Float> {
    val flatArray = rows.flatMap { it.asIterable() }.toFloatArray()
    val byteData = ByteArrayConverter.convertFloatArrayToBytes(flatArray)
    return fromBytes(FP32, shape, byteData) as Tensor<FP32, Float>
}


// ==============================================
// Int Array Extensions (Int32)
// ==============================================

/**
 * Creates a tensor from Int varargs (vector format).
 * Usage: fromArray(1, -1, 2, -2)
 */
public fun WeightsScope<Int32, Int>.fromArray(vararg values: Int): Tensor<Int32, Int> {
    val intArray = values
    val byteData = ByteArrayConverter.convertIntArrayToBytes(intArray)
    return fromBytes(Int32, shape, byteData) as Tensor<Int32, Int>
}

public fun BiasScope<Int32, Int>.fromArray(vararg values: Int): Tensor<Int32, Int> {
    val intArray = values
    val byteData = ByteArrayConverter.convertIntArrayToBytes(intArray)
    return fromBytes(Int32, shape, byteData) as Tensor<Int32, Int>
}

/**
 * Creates a tensor from Int array (explicit array).
 * Usage: fromIntArray(intArrayOf(1, -1, 2))
 */
public fun WeightsScope<Int32, Int>.fromIntArray(values: IntArray): Tensor<Int32, Int> {
    val byteData = ByteArrayConverter.convertIntArrayToBytes(values)
    return fromBytes(Int32, shape, byteData) as Tensor<Int32, Int>
}

public fun BiasScope<Int32, Int>.fromIntArray(values: IntArray): Tensor<Int32, Int> {
    val byteData = ByteArrayConverter.convertIntArrayToBytes(values)
    return fromBytes(Int32, shape, byteData) as Tensor<Int32, Int>
}

/**
 * Creates a tensor from Int matrix format (List of Lists).
 */
public fun WeightsScope<Int32, Int>.fromIntMatrix(vararg rows: List<Int>): Tensor<Int32, Int> {
    val flatArray = rows.flatMap { it }.toIntArray()
    val byteData = ByteArrayConverter.convertIntArrayToBytes(flatArray)
    return fromBytes(Int32, shape, byteData) as Tensor<Int32, Int>
}

public fun BiasScope<Int32, Int>.fromIntMatrix(vararg rows: List<Int>): Tensor<Int32, Int> {
    val flatArray = rows.flatMap { it }.toIntArray()
    val byteData = ByteArrayConverter.convertIntArrayToBytes(flatArray)
    return fromBytes(Int32, shape, byteData) as Tensor<Int32, Int>
}

/**
 * Creates a tensor from Int matrix format (Array of Arrays).
 */
public fun WeightsScope<Int32, Int>.fromIntArrayMatrix(vararg rows: IntArray): Tensor<Int32, Int> {
    val flatArray = rows.flatMap { it.asIterable() }.toIntArray()
    val byteData = ByteArrayConverter.convertIntArrayToBytes(flatArray)
    return fromBytes(Int32, shape, byteData) as Tensor<Int32, Int>
}

public fun BiasScope<Int32, Int>.fromIntArrayMatrix(vararg rows: IntArray): Tensor<Int32, Int> {
    val flatArray = rows.flatMap { it.asIterable() }.toIntArray()
    val byteData = ByteArrayConverter.convertIntArrayToBytes(flatArray)
    return fromBytes(Int32, shape, byteData) as Tensor<Int32, Int>
}