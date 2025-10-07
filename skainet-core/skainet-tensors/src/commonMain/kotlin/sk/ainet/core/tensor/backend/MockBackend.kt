package sk.ainet.core.tensor.backend

import sk.ainet.core.tensor.*

/**
 * Mock backend implementations that throw NotImplementedError for all operations.
 * These backends enable tensor creation with real data (e.g., from files) while
 * keeping operations unimplemented, allowing for separation between data handling
 * and computation logic.
 */

/**
 * Mock backend for FP32/Float tensors that throws NotImplementedError for all operations.
 */
public class MockBackendFP32 : ComputeBackend<FP32, Float, Tensor<FP32, Float>> {
    override val name: String = "MockBackendFP32"

    override fun matmul(a: Tensor<FP32, Float>, b: Tensor<FP32, Float>): Tensor<FP32, Float> {
        throw NotImplementedError("Matrix multiplication operation is not implemented in mock backend")
    }

    override fun matmul4d(a: Tensor<FP32, Float>, b: Tensor<FP32, Float>): Tensor<FP32, Float> {
        throw NotImplementedError("4D matrix multiplication operation is not implemented in mock backend")
    }

    override fun scale(a: Tensor<FP32, Float>, scalar: Double): Tensor<FP32, Float> {
        throw NotImplementedError("Scale operation is not implemented in mock backend")
    }

    override fun dot(a: Tensor<FP32, Float>, b: Tensor<FP32, Float>): Double {
        throw NotImplementedError("Dot product operation is not implemented in mock backend")
    }

    // Tensor-Tensor operations
    override fun Tensor<FP32, Float>.plus(other: Tensor<FP32, Float>): Tensor<FP32, Float> {
        throw NotImplementedError("Tensor addition operation is not implemented in mock backend")
    }

    override fun Tensor<FP32, Float>.minus(other: Tensor<FP32, Float>): Tensor<FP32, Float> {
        throw NotImplementedError("Tensor subtraction operation is not implemented in mock backend")
    }

    override fun Tensor<FP32, Float>.times(other: Tensor<FP32, Float>): Tensor<FP32, Float> {
        throw NotImplementedError("Element-wise tensor multiplication operation is not implemented in mock backend")
    }

    override fun Tensor<FP32, Float>.div(other: Tensor<FP32, Float>): Tensor<FP32, Float> {
        throw NotImplementedError("Element-wise tensor division operation is not implemented in mock backend")
    }

    // Tensor-Scalar operations
    override fun Tensor<FP32, Float>.plus(scalar: Int): Tensor<FP32, Float> {
        throw NotImplementedError("Tensor-Int addition operation is not implemented in mock backend")
    }

    override fun Tensor<FP32, Float>.minus(scalar: Int): Tensor<FP32, Float> {
        throw NotImplementedError("Tensor-Int subtraction operation is not implemented in mock backend")
    }

    override fun Tensor<FP32, Float>.times(scalar: Int): Tensor<FP32, Float> {
        throw NotImplementedError("Tensor-Int multiplication operation is not implemented in mock backend")
    }

    override fun Tensor<FP32, Float>.div(scalar: Int): Tensor<FP32, Float> {
        throw NotImplementedError("Tensor-Int division operation is not implemented in mock backend")
    }

    override fun Tensor<FP32, Float>.plus(scalar: Float): Tensor<FP32, Float> {
        throw NotImplementedError("Tensor-Float addition operation is not implemented in mock backend")
    }

    override fun Tensor<FP32, Float>.minus(scalar: Float): Tensor<FP32, Float> {
        throw NotImplementedError("Tensor-Float subtraction operation is not implemented in mock backend")
    }

    override fun Tensor<FP32, Float>.times(scalar: Float): Tensor<FP32, Float> {
        throw NotImplementedError("Tensor-Float multiplication operation is not implemented in mock backend")
    }

    override fun Tensor<FP32, Float>.div(scalar: Float): Tensor<FP32, Float> {
        throw NotImplementedError("Tensor-Float division operation is not implemented in mock backend")
    }

    override fun Tensor<FP32, Float>.plus(scalar: Double): Tensor<FP32, Float> {
        throw NotImplementedError("Tensor-Double addition operation is not implemented in mock backend")
    }

    override fun Tensor<FP32, Float>.minus(scalar: Double): Tensor<FP32, Float> {
        throw NotImplementedError("Tensor-Double subtraction operation is not implemented in mock backend")
    }

    override fun Tensor<FP32, Float>.times(scalar: Double): Tensor<FP32, Float> {
        throw NotImplementedError("Tensor-Double multiplication operation is not implemented in mock backend")
    }

    override fun Tensor<FP32, Float>.div(scalar: Double): Tensor<FP32, Float> {
        throw NotImplementedError("Tensor-Double division operation is not implemented in mock backend")
    }

    // Scalar-Tensor operations  
    override fun Double.plus(t: Tensor<FP32, Float>): Tensor<FP32, Float> {
        throw NotImplementedError("Double-Tensor addition operation is not implemented in mock backend")
    }

    override fun Double.minus(t: Tensor<FP32, Float>): Tensor<FP32, Float> {
        throw NotImplementedError("Double-Tensor subtraction operation is not implemented in mock backend")
    }

    override fun Double.times(t: Tensor<FP32, Float>): Tensor<FP32, Float> {
        throw NotImplementedError("Double-Tensor multiplication operation is not implemented in mock backend")
    }

    override fun Double.div(t: Tensor<FP32, Float>): Tensor<FP32, Float> {
        throw NotImplementedError("Double-Tensor division operation is not implemented in mock backend")
    }

    // Mathematical functions
    override fun Tensor<FP32, Float>.t(): Tensor<FP32, Float> {
        throw NotImplementedError("Transpose operation is not implemented in mock backend")
    }

    override fun Tensor<FP32, Float>.relu(): Tensor<FP32, Float> {
        throw NotImplementedError("ReLU operation is not implemented in mock backend")
    }

    override fun Tensor<FP32, Float>.softmax(dimension: Int): Tensor<FP32, Float> {
        throw NotImplementedError("Softmax operation is not implemented in mock backend")
    }

    override fun Tensor<FP32, Float>.sigmoid(): Tensor<FP32, Float> {
        throw NotImplementedError("Sigmoid operation is not implemented in mock backend")
    }

    override fun Tensor<FP32, Float>.tanh(): Tensor<FP32, Float> {
        throw NotImplementedError("Tanh operation is not implemented in mock backend")
    }

    // Shape operations
    override fun Tensor<FP32, Float>.flatten(startDim: Int, endDim: Int): Tensor<FP32, Float> {
        throw NotImplementedError("Flatten operation is not implemented in mock backend")
    }

    override fun Tensor<FP32, Float>.reshape(newShape: Shape): Tensor<FP32, Float> {
        throw NotImplementedError("Reshape operation is not implemented in mock backend")
    }

    override fun Tensor<FP32, Float>.reshape(vararg dimensions: Int): Tensor<FP32, Float> {
        throw NotImplementedError("Reshape operation is not implemented in mock backend")
    }
}

/**
 * Mock backend for Int32/Int tensors that throws NotImplementedError for all operations.
 */
public class MockBackendInt32 : ComputeBackend<Int32, Int, Tensor<Int32, Int>> {
    override val name: String = "MockBackendInt32"

    override fun matmul(a: Tensor<Int32, Int>, b: Tensor<Int32, Int>): Tensor<Int32, Int> {
        throw NotImplementedError("Matrix multiplication operation is not implemented in mock backend")
    }

    override fun matmul4d(a: Tensor<Int32, Int>, b: Tensor<Int32, Int>): Tensor<Int32, Int> {
        throw NotImplementedError("4D matrix multiplication operation is not implemented in mock backend")
    }

    override fun scale(a: Tensor<Int32, Int>, scalar: Double): Tensor<Int32, Int> {
        throw NotImplementedError("Scale operation is not implemented in mock backend")
    }

    override fun dot(a: Tensor<Int32, Int>, b: Tensor<Int32, Int>): Double {
        throw NotImplementedError("Dot product operation is not implemented in mock backend")
    }

    // Tensor-Tensor operations
    override fun Tensor<Int32, Int>.plus(other: Tensor<Int32, Int>): Tensor<Int32, Int> {
        throw NotImplementedError("Tensor addition operation is not implemented in mock backend")
    }

    override fun Tensor<Int32, Int>.minus(other: Tensor<Int32, Int>): Tensor<Int32, Int> {
        throw NotImplementedError("Tensor subtraction operation is not implemented in mock backend")
    }

    override fun Tensor<Int32, Int>.times(other: Tensor<Int32, Int>): Tensor<Int32, Int> {
        throw NotImplementedError("Element-wise tensor multiplication operation is not implemented in mock backend")
    }

    override fun Tensor<Int32, Int>.div(other: Tensor<Int32, Int>): Tensor<Int32, Int> {
        throw NotImplementedError("Element-wise tensor division operation is not implemented in mock backend")
    }

    // Tensor-Scalar operations
    override fun Tensor<Int32, Int>.plus(scalar: Int): Tensor<Int32, Int> {
        throw NotImplementedError("Tensor-Int addition operation is not implemented in mock backend")
    }

    override fun Tensor<Int32, Int>.minus(scalar: Int): Tensor<Int32, Int> {
        throw NotImplementedError("Tensor-Int subtraction operation is not implemented in mock backend")
    }

    override fun Tensor<Int32, Int>.times(scalar: Int): Tensor<Int32, Int> {
        throw NotImplementedError("Tensor-Int multiplication operation is not implemented in mock backend")
    }

    override fun Tensor<Int32, Int>.div(scalar: Int): Tensor<Int32, Int> {
        throw NotImplementedError("Tensor-Int division operation is not implemented in mock backend")
    }

    override fun Tensor<Int32, Int>.plus(scalar: Float): Tensor<Int32, Int> {
        throw NotImplementedError("Tensor-Float addition operation is not implemented in mock backend")
    }

    override fun Tensor<Int32, Int>.minus(scalar: Float): Tensor<Int32, Int> {
        throw NotImplementedError("Tensor-Float subtraction operation is not implemented in mock backend")
    }

    override fun Tensor<Int32, Int>.times(scalar: Float): Tensor<Int32, Int> {
        throw NotImplementedError("Tensor-Float multiplication operation is not implemented in mock backend")
    }

    override fun Tensor<Int32, Int>.div(scalar: Float): Tensor<Int32, Int> {
        throw NotImplementedError("Tensor-Float division operation is not implemented in mock backend")
    }

    override fun Tensor<Int32, Int>.plus(scalar: Double): Tensor<Int32, Int> {
        throw NotImplementedError("Tensor-Double addition operation is not implemented in mock backend")
    }

    override fun Tensor<Int32, Int>.minus(scalar: Double): Tensor<Int32, Int> {
        throw NotImplementedError("Tensor-Double subtraction operation is not implemented in mock backend")
    }

    override fun Tensor<Int32, Int>.times(scalar: Double): Tensor<Int32, Int> {
        throw NotImplementedError("Tensor-Double multiplication operation is not implemented in mock backend")
    }

    override fun Tensor<Int32, Int>.div(scalar: Double): Tensor<Int32, Int> {
        throw NotImplementedError("Tensor-Double division operation is not implemented in mock backend")
    }

    // Scalar-Tensor operations  
    override fun Double.plus(t: Tensor<Int32, Int>): Tensor<Int32, Int> {
        throw NotImplementedError("Double-Tensor addition operation is not implemented in mock backend")
    }

    override fun Double.minus(t: Tensor<Int32, Int>): Tensor<Int32, Int> {
        throw NotImplementedError("Double-Tensor subtraction operation is not implemented in mock backend")
    }

    override fun Double.times(t: Tensor<Int32, Int>): Tensor<Int32, Int> {
        throw NotImplementedError("Double-Tensor multiplication operation is not implemented in mock backend")
    }

    override fun Double.div(t: Tensor<Int32, Int>): Tensor<Int32, Int> {
        throw NotImplementedError("Double-Tensor division operation is not implemented in mock backend")
    }

    // Mathematical functions
    override fun Tensor<Int32, Int>.t(): Tensor<Int32, Int> {
        throw NotImplementedError("Transpose operation is not implemented in mock backend")
    }

    override fun Tensor<Int32, Int>.relu(): Tensor<Int32, Int> {
        throw NotImplementedError("ReLU operation is not implemented in mock backend")
    }

    override fun Tensor<Int32, Int>.softmax(dimension: Int): Tensor<Int32, Int> {
        throw NotImplementedError("Softmax operation is not implemented in mock backend")
    }

    override fun Tensor<Int32, Int>.sigmoid(): Tensor<Int32, Int> {
        throw NotImplementedError("Sigmoid operation is not implemented in mock backend")
    }

    override fun Tensor<Int32, Int>.tanh(): Tensor<Int32, Int> {
        throw NotImplementedError("Tanh operation is not implemented in mock backend")
    }

    // Shape operations
    override fun Tensor<Int32, Int>.flatten(startDim: Int, endDim: Int): Tensor<Int32, Int> {
        throw NotImplementedError("Flatten operation is not implemented in mock backend")
    }

    override fun Tensor<Int32, Int>.reshape(newShape: Shape): Tensor<Int32, Int> {
        throw NotImplementedError("Reshape operation is not implemented in mock backend")
    }

    override fun Tensor<Int32, Int>.reshape(vararg dimensions: Int): Tensor<Int32, Int> {
        throw NotImplementedError("Reshape operation is not implemented in mock backend")
    }
}

/**
 * Mock backend for Int8/Byte tensors that throws NotImplementedError for all operations.
 */
public class MockBackendInt8 : ComputeBackend<Int8, Byte, Tensor<Int8, Byte>> {
    override val name: String = "MockBackendInt8"

    override fun matmul(a: Tensor<Int8, Byte>, b: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        throw NotImplementedError("Matrix multiplication operation is not implemented in mock backend")
    }

    override fun matmul4d(a: Tensor<Int8, Byte>, b: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        throw NotImplementedError("4D matrix multiplication operation is not implemented in mock backend")
    }

    override fun scale(a: Tensor<Int8, Byte>, scalar: Double): Tensor<Int8, Byte> {
        throw NotImplementedError("Scale operation is not implemented in mock backend")
    }

    override fun dot(a: Tensor<Int8, Byte>, b: Tensor<Int8, Byte>): Double {
        throw NotImplementedError("Dot product operation is not implemented in mock backend")
    }

    // Tensor-Tensor operations
    override fun Tensor<Int8, Byte>.plus(other: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        throw NotImplementedError("Tensor addition operation is not implemented in mock backend")
    }

    override fun Tensor<Int8, Byte>.minus(other: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        throw NotImplementedError("Tensor subtraction operation is not implemented in mock backend")
    }

    override fun Tensor<Int8, Byte>.times(other: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        throw NotImplementedError("Element-wise tensor multiplication operation is not implemented in mock backend")
    }

    override fun Tensor<Int8, Byte>.div(other: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        throw NotImplementedError("Element-wise tensor division operation is not implemented in mock backend")
    }

    // Tensor-Scalar operations
    override fun Tensor<Int8, Byte>.plus(scalar: Int): Tensor<Int8, Byte> {
        throw NotImplementedError("Tensor-Int addition operation is not implemented in mock backend")
    }

    override fun Tensor<Int8, Byte>.minus(scalar: Int): Tensor<Int8, Byte> {
        throw NotImplementedError("Tensor-Int subtraction operation is not implemented in mock backend")
    }

    override fun Tensor<Int8, Byte>.times(scalar: Int): Tensor<Int8, Byte> {
        throw NotImplementedError("Tensor-Int multiplication operation is not implemented in mock backend")
    }

    override fun Tensor<Int8, Byte>.div(scalar: Int): Tensor<Int8, Byte> {
        throw NotImplementedError("Tensor-Int division operation is not implemented in mock backend")
    }

    override fun Tensor<Int8, Byte>.plus(scalar: Float): Tensor<Int8, Byte> {
        throw NotImplementedError("Tensor-Float addition operation is not implemented in mock backend")
    }

    override fun Tensor<Int8, Byte>.minus(scalar: Float): Tensor<Int8, Byte> {
        throw NotImplementedError("Tensor-Float subtraction operation is not implemented in mock backend")
    }

    override fun Tensor<Int8, Byte>.times(scalar: Float): Tensor<Int8, Byte> {
        throw NotImplementedError("Tensor-Float multiplication operation is not implemented in mock backend")
    }

    override fun Tensor<Int8, Byte>.div(scalar: Float): Tensor<Int8, Byte> {
        throw NotImplementedError("Tensor-Float division operation is not implemented in mock backend")
    }

    override fun Tensor<Int8, Byte>.plus(scalar: Double): Tensor<Int8, Byte> {
        throw NotImplementedError("Tensor-Double addition operation is not implemented in mock backend")
    }

    override fun Tensor<Int8, Byte>.minus(scalar: Double): Tensor<Int8, Byte> {
        throw NotImplementedError("Tensor-Double subtraction operation is not implemented in mock backend")
    }

    override fun Tensor<Int8, Byte>.times(scalar: Double): Tensor<Int8, Byte> {
        throw NotImplementedError("Tensor-Double multiplication operation is not implemented in mock backend")
    }

    override fun Tensor<Int8, Byte>.div(scalar: Double): Tensor<Int8, Byte> {
        throw NotImplementedError("Tensor-Double division operation is not implemented in mock backend")
    }

    // Scalar-Tensor operations  
    override fun Double.plus(t: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        throw NotImplementedError("Double-Tensor addition operation is not implemented in mock backend")
    }

    override fun Double.minus(t: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        throw NotImplementedError("Double-Tensor subtraction operation is not implemented in mock backend")
    }

    override fun Double.times(t: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        throw NotImplementedError("Double-Tensor multiplication operation is not implemented in mock backend")
    }

    override fun Double.div(t: Tensor<Int8, Byte>): Tensor<Int8, Byte> {
        throw NotImplementedError("Double-Tensor division operation is not implemented in mock backend")
    }

    // Mathematical functions
    override fun Tensor<Int8, Byte>.t(): Tensor<Int8, Byte> {
        throw NotImplementedError("Transpose operation is not implemented in mock backend")
    }

    override fun Tensor<Int8, Byte>.relu(): Tensor<Int8, Byte> {
        throw NotImplementedError("ReLU operation is not implemented in mock backend")
    }

    override fun Tensor<Int8, Byte>.softmax(dimension: Int): Tensor<Int8, Byte> {
        throw NotImplementedError("Softmax operation is not implemented in mock backend")
    }

    override fun Tensor<Int8, Byte>.sigmoid(): Tensor<Int8, Byte> {
        throw NotImplementedError("Sigmoid operation is not implemented in mock backend")
    }

    override fun Tensor<Int8, Byte>.tanh(): Tensor<Int8, Byte> {
        throw NotImplementedError("Tanh operation is not implemented in mock backend")
    }

    // Shape operations
    override fun Tensor<Int8, Byte>.flatten(startDim: Int, endDim: Int): Tensor<Int8, Byte> {
        throw NotImplementedError("Flatten operation is not implemented in mock backend")
    }

    override fun Tensor<Int8, Byte>.reshape(newShape: Shape): Tensor<Int8, Byte> {
        throw NotImplementedError("Reshape operation is not implemented in mock backend")
    }

    override fun Tensor<Int8, Byte>.reshape(vararg dimensions: Int): Tensor<Int8, Byte> {
        throw NotImplementedError("Reshape operation is not implemented in mock backend")
    }
}