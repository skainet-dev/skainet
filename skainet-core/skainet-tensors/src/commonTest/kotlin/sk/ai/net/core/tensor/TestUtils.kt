package sk.ai.net.core.tensor

import kotlin.test.assertTrue
import kotlin.math.abs

/**
 * Utility functions for testing with float precision handling
 */

/**
 * Asserts that two float values are equal within a specified tolerance/precision
 */
fun assertFloatEquals(expected: Float, actual: Float, precision: Float = 1e-6f, message: String? = null) {
    val diff = abs(expected - actual)
    val errorMessage = message ?: "Expected $expected but was $actual (difference: $diff, tolerance: $precision)"
    assertTrue(diff <= precision, errorMessage)
}

/**
 * Asserts that two double values are equal within a specified tolerance/precision
 */
fun assertDoubleEquals(expected: Double, actual: Double, precision: Double = 1e-6, message: String? = null) {
    val diff = abs(expected - actual)
    val errorMessage = message ?: "Expected $expected but was $actual (difference: $diff, tolerance: $precision)"
    assertTrue(diff <= precision, errorMessage)
}

/**
 * Asserts that a float value is within a relative tolerance of the expected value
 * Useful for large numbers where absolute tolerance might not be appropriate
 */
fun assertFloatEqualsRelative(expected: Float, actual: Float, relativeTolerance: Float = 1e-5f, message: String? = null) {
    val relativeError = abs((expected - actual) / expected)
    val errorMessage = message ?: "Expected $expected but was $actual (relative error: $relativeError, tolerance: $relativeTolerance)"
    assertTrue(relativeError <= relativeTolerance, errorMessage)
}

/**
 * Asserts that a double value is within a relative tolerance of the expected value
 */
fun assertDoubleEqualsRelative(expected: Double, actual: Double, relativeTolerance: Double = 1e-5, message: String? = null) {
    val relativeError = abs((expected - actual) / expected)
    val errorMessage = message ?: "Expected $expected but was $actual (relative error: $relativeError, tolerance: $relativeTolerance)"
    assertTrue(relativeError <= relativeTolerance, errorMessage)
}