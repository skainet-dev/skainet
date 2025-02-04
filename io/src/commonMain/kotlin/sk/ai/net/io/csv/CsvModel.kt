package sk.ai.net.io.csv

import kotlinx.serialization.Serializable

@Serializable
data class ArrayValues(
    val unique_parameter_name: String,
    val array_values: ArrayValue
)

@Serializable
data class ArrayValue(
    val values: List<Double>,
    val shape: List<Int>
)

@Serializable
data class Tensor(
    val shape: List<Int>,
    val values: List<Double>
)

@Serializable
data class Parameter(
    val unique_parameter_name: String,
    val tensor: Tensor
)