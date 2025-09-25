package sk.ainet.core.tensor


public data class Slice<T : DType, V>(
    val tensor: Tensor<T, V>,
    val dimensionIndex: Int,
    val startIndex: Long,
    val endIndex: Long
) {
    public fun toRange(): LongRange = startIndex..endIndex // inclusive
}

// Option A: keep the type params
public fun <T : DType, V> Slice<T, V>.start(): Long = startIndex
public fun <T : DType, V> Slice<T, V>.end(): Long = endIndex
public fun <T : DType, V> Slice<T, V>.all(): LongRange {
    val last = tensor.shape.dimensions[dimensionIndex].toLong() - 1 // last valid index
    return startIndex..last
}
public fun <T : DType, V> Slice<T, V>.range(): LongRange = startIndex..endIndex
