package sk.ainet.core.tensor.backend

import sk.ainet.core.tensor.*

/**
 * Generic CPU-based tensor implementation using composition pattern.
 *
 * This unified tensor class can work with any DType and value type combination,
 * providing a single implementation that replaces the specific CpuTensorFP32,
 * CpuTensorInt32, CpuTensorInt8, etc. implementations.
 *
 * @param T The DType (data type) such as FP32, Int32, Int8, FP16, Int4, Ternary
 * @param V The value type such as Float, Int, Byte, etc.
 */
public class CpuTensor<T : DType, V>(
    private val tensorData: TensorData<T, V>,
    private val backend: ComputeBackend<T, V, Tensor<T, V>>
) : Tensor<T, V> {

    // Implement composition pattern properties
    override val data: TensorData<T, V> = tensorData
    override val ops: TensorOps<T, V, Tensor<T, V>> = backend
    override val shape: Shape get() = data.shape

    init {
        require(data.shape.rank in 1..4) {
            "Only 1-4 dimensional tensors are supported, got ${data.shape.rank}"
        }
    }

    override fun toString(): String = "CpuTensor<${backend.name}>(${shape})"

    public companion object {
        /**
         * Creates a generic tensor from tensor data and backend.
         */
        public fun <T : DType, V> create(
            data: TensorData<T, V>,
            backend: ComputeBackend<T, V, Tensor<T, V>>
        ): CpuTensor<T, V> {
            return CpuTensor(data, backend)
        }

        /**
         * Creates a tensor with the given shape and data array, using the appropriate backend.
         */
        public inline fun <reified T : DType, reified V> fromArray(
            shape: Shape,
            data: Array<V>,
            backend: ComputeBackend<T, V, Tensor<T, V>>
        ): CpuTensor<T, V> {
            require(data.size == shape.volume) {
                "Data size ${data.size} doesn't match shape volume ${shape.volume}"
            }
            val tensorData = DenseTensorData<T, V>(shape, data)
            return CpuTensor(tensorData, backend)
        }

        /**
         * Creates a tensor filled with zeros using a default value array.
         * Note: This is a basic implementation. Specific backends may provide optimized versions.
         */
        public inline fun <reified T : DType, reified V> zeros(
            shape: Shape,
            backend: ComputeBackend<T, V, Tensor<T, V>>,
            defaultValue: V
        ): CpuTensor<T, V> {
            val data = Array<V>(shape.volume) { defaultValue }
            val tensorData = DenseTensorData<T, V>(shape, data)
            return CpuTensor(tensorData, backend)
        }
    }
}