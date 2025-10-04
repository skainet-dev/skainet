package sk.ainet.core.tensor

import kotlin.random.Random

public interface TensorFactory<T : DType, V> {
    public fun zeros(shape: Shape): Tensor<T, V>
    public fun ones(shape: Shape): Tensor<T, V>
    public fun random(shape: Shape): Tensor<T, V>

    // Advanced random methods with seed control
    public fun random(shape: Shape, seed: Long): Tensor<T, V>
    public fun random(shape: Shape, random: Random): Tensor<T, V>

    // Distribution-based random methods
    public fun randomNormal(shape: Shape, mean: Double = 0.0, std: Double = 1.0): Tensor<T, V>
    public fun randomNormal(shape: Shape, mean: Double = 0.0, std: Double = 1.0, seed: Long): Tensor<T, V>
    public fun randomNormal(shape: Shape, mean: Double = 0.0, std: Double = 1.0, random: Random): Tensor<T, V>

    public fun randomUniform(shape: Shape, min: Double = 0.0, max: Double = 1.0): Tensor<T, V>
    public fun randomUniform(shape: Shape, min: Double = 0.0, max: Double = 1.0, seed: Long): Tensor<T, V>
    public fun randomUniform(shape: Shape, min: Double = 0.0, max: Double = 1.0, random: Random): Tensor<T, V>

    public fun fromArray(shape: Shape, data: FloatArray): Tensor<T, V>
    public fun fromArray(shape: Shape, data: IntArray): Tensor<T, V>
}