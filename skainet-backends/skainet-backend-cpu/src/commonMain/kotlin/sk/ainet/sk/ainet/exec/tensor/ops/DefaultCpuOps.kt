package sk.ainet.sk.ainet.exec.tensor.ops

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.DType
import sk.ainet.lang.ops.TensorOp
import sk.ainet.lang.ops.InProgress
import sk.ainet.lang.tensor.data.TensorDataFactory

@InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#defaultcpuops")
public class DefaultCpuOps<V>(private val dataFactory: TensorDataFactory) : TensorOps<V> {

    private class CpuTensor<T : DType, V>(
        override val data: sk.ainet.lang.tensor.data.TensorData<T, V>,
        private val opsRef: TensorOps<V>,
        override val dtype: kotlin.reflect.KClass<T>
    ) : Tensor<T, V> {
        override val ops: TensorOps<V>
            get() = opsRef
    }

    private fun broadcastShapes(a: Shape, b: Shape): Shape {
        val ad = a.dimensions
        val bd = b.dimensions
        val maxRank = maxOf(ad.size, bd.size)
        val out = IntArray(maxRank)
        var ai = ad.size - 1
        var bi = bd.size - 1
        for (oi in maxRank - 1 downTo 0) {
            val asz = if (ai >= 0) ad[ai] else 1
            val bsz = if (bi >= 0) bd[bi] else 1
            if (asz != bsz && asz != 1 && bsz != 1) {
                throw IllegalArgumentException("Shapes ${a.dimensions.contentToString()} and ${b.dimensions.contentToString()} cannot be broadcasted")
            }
            out[oi] = maxOf(asz, bsz)
            ai--; bi--
        }
        return Shape(out)
    }

    private fun mapIndex(idx: IntArray, inShape: Shape): IntArray {
        // Map output index to input index with broadcasting: if input dim == 1, use 0 for that dim.
        val inDims = inShape.dimensions
        val outRank = idx.size
        val inRank = inDims.size
        val mapped = IntArray(inRank)
        var ir = inRank - 1
        var or = outRank - 1
        while (ir >= 0) {
            val inDim = inDims[ir]
            val outIndex = if (or >= 0) idx[or] else 0
            mapped[ir] = if (inDim == 1) 0 else outIndex
            ir--; or--
        }
        return mapped
    }

    private fun <T : DType> requireSameDType(a: Tensor<T, V>, b: Tensor<T, V>) {
        require(a.dtype == b.dtype) { "DType mismatch: ${'$'}{a.dtype} vs ${'$'}{b.dtype}" }
    }

    private fun <T : DType> elementwise(
        a: Tensor<T, V>,
        b: Tensor<T, V>,
        op: (av: V, bv: V, dtype: kotlin.reflect.KClass<T>) -> V
    ): Tensor<T, V> {
        requireSameDType(a, b)
        val outShape = broadcastShapes(a.shape, b.shape)
        val outData = dataFactory.init<T, V>(outShape, a.dtype) { outIdx ->
            val ai = mapIndex(outIdx, a.shape)
            val bi = mapIndex(outIdx, b.shape)
            val av = a.data.get(*ai)
            val bv = b.data.get(*bi)
            op(av, bv, a.dtype)
        }
        return CpuTensor(outData, this, a.dtype)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-add")
    override fun <T : DType> add(
        a: Tensor<T, V>,
        b: Tensor<T, V>
    ): Tensor<T, V> {
        return elementwise(a, b) { av, bv, dtype ->
            when (dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val x = av as Float;
                    val y = bv as Float; (x + y) as V
                }

                sk.ainet.lang.types.Int32::class -> {
                    val x = av as Int;
                    val y = bv as Int; (x + y) as V
                }

                else -> throw IllegalArgumentException("Unsupported dtype for add: ${'$'}dtype")
            }
        }
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-subtract")
    override fun <T : DType> subtract(
        a: Tensor<T, V>,
        b: Tensor<T, V>
    ): Tensor<T, V> {
        return elementwise(a, b) { av, bv, dtype ->
            when (dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val x = av as Float;
                    val y = bv as Float; (x - y) as V
                }

                sk.ainet.lang.types.Int32::class -> {
                    val x = av as Int;
                    val y = bv as Int; (x - y) as V
                }

                else -> throw IllegalArgumentException("Unsupported dtype for subtract: ${'$'}dtype")
            }
        }
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-multiply")
    override fun <T : DType> multiply(
        a: Tensor<T, V>,
        b: Tensor<T, V>
    ): Tensor<T, V> {
        return elementwise(a, b) { av, bv, dtype ->
            when (dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val x = av as Float;
                    val y = bv as Float; (x * y) as V
                }

                sk.ainet.lang.types.Int32::class -> {
                    val x = av as Int;
                    val y = bv as Int; (x * y) as V
                }

                else -> throw IllegalArgumentException("Unsupported dtype for multiply: ${'$'}dtype")
            }
        }
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-divide")
    override fun <T : DType> divide(
        a: Tensor<T, V>,
        b: Tensor<T, V>
    ): Tensor<T, V> {
        return elementwise(a, b) { av, bv, dtype ->
            when (dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val x = av as Float;
                    val y = bv as Float; (x / y) as V
                }

                sk.ainet.lang.types.Int32::class -> {
                    val x = av as Int;
                    val y = bv as Int; if (y == 0) 0 as V else (x / y) as V
                }

                else -> throw IllegalArgumentException("Unsupported dtype for divide: ${'$'}dtype")
            }
        }
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-matmul")
    override fun <T : DType> matmul(
        a: Tensor<T, V>,
        b: Tensor<T, V>
    ): Tensor<T, V> {
        require(a.rank >= 2 && b.rank >= 2) { "Matrix multiplication requires tensors with at least 2 dimensions" }
        require(a.dtype == b.dtype) { "DType mismatch: ${a.dtype} vs ${b.dtype}" }

        val aDims = a.shape.dimensions
        val bDims = b.shape.dimensions
        val aRank = aDims.size
        val bRank = bDims.size
        val kA = aDims[aRank - 1]
        val kB = bDims[bRank - 2]
        require(kA == kB) { "Matrix multiplication shape mismatch: inner dimensions must match ($kA vs $kB)" }

        // Validate batch dims broadcastability (excluding last two dims)
        val maxRank = maxOf(aRank, bRank)
        for (i in 0 until maxRank - 2) {
            val aDim = if (i < aRank - 2) aDims[i] else 1
            val bDim = if (i < bRank - 2) bDims[i] else 1
            if (aDim != bDim && aDim != 1 && bDim != 1) {
                throw IllegalArgumentException("Matrix multiplication batch dimension mismatch at position $i: $aDim vs $bDim")
            }
        }

        // Compute output shape
        val outDims = IntArray(maxRank)
        for (i in 0 until maxRank - 2) {
            val aDim = if (i < aRank - 2) aDims[i] else 1
            val bDim = if (i < bRank - 2) bDims[i] else 1
            outDims[i] = maxOf(aDim, bDim)
        }
        outDims[maxRank - 2] = aDims[aRank - 2] // m
        outDims[maxRank - 1] = bDims[bRank - 1] // n
        val outShape = Shape(outDims)

        // Helper to map an output batch index to input batch indices with broadcasting
        fun mapBatchIndex(batchIdx: IntArray, inDims: IntArray, inRank: Int): IntArray {
            val inBatchRank = inRank - 2
            val mapped = IntArray(inBatchRank)
            var or = batchIdx.size - 1
            var ir = inBatchRank - 1
            while (ir >= 0) {
                val inDim = inDims[ir]
                val outIndex = if (or >= 0) batchIdx[or] else 0
                mapped[ir] = if (inDim == 1) 0 else outIndex
                ir--; or--
            }
            return mapped
        }

        val outData = dataFactory.init<T, V>(outShape, a.dtype) { outIdx ->
            // Split outIdx into batch + m + n
            val m = outIdx[outIdx.size - 2]
            val n = outIdx[outIdx.size - 1]
            val batchIdx = if (outIdx.size > 2) outIdx.copyOf(outIdx.size - 2) else IntArray(0)

            val aBatchIdx = mapBatchIndex(batchIdx, aDims, aRank)
            val bBatchIdx = mapBatchIndex(batchIdx, bDims, bRank)

            // Accumulate over k
            when (a.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    var acc = 0.0f
                    var k = 0
                    while (k < kA) {
                        // Build indices for a: [aBatch..., m, k]
                        val aIdx = IntArray(aRank)
                        if (aBatchIdx.isNotEmpty()) {
                            aBatchIdx.copyInto(aIdx, destinationOffset = 0, startIndex = 0, endIndex = aBatchIdx.size)
                        }
                        aIdx[aRank - 2] = m
                        aIdx[aRank - 1] = k
                        val av = a.data.get(*aIdx) as Float

                        // Build indices for b: [bBatch..., k, n]
                        val bIdx = IntArray(bRank)
                        if (bBatchIdx.isNotEmpty()) {
                            bBatchIdx.copyInto(bIdx, destinationOffset = 0, startIndex = 0, endIndex = bBatchIdx.size)
                        }
                        bIdx[bRank - 2] = k
                        bIdx[bRank - 1] = n
                        val bv = b.data.get(*bIdx) as Float

                        acc += av * bv
                        k++
                    }
                    @Suppress("UNCHECKED_CAST")
                    acc as V
                }

                sk.ainet.lang.types.Int32::class,
                sk.ainet.lang.types.Int8::class -> {
                    var acc = 0
                    var k = 0
                    while (k < kA) {
                        val aIdx = IntArray(aRank)
                        if (aBatchIdx.isNotEmpty()) {
                            aBatchIdx.copyInto(aIdx, destinationOffset = 0, startIndex = 0, endIndex = aBatchIdx.size)
                        }
                        aIdx[aRank - 2] = m
                        aIdx[aRank - 1] = k
                        val av = a.data.get(*aIdx) as Int

                        val bIdx = IntArray(bRank)
                        if (bBatchIdx.isNotEmpty()) {
                            bBatchIdx.copyInto(bIdx, destinationOffset = 0, startIndex = 0, endIndex = bBatchIdx.size)
                        }
                        bIdx[bRank - 2] = k
                        bIdx[bRank - 1] = n
                        val bv = b.data.get(*bIdx) as Int

                        acc += av * bv
                        k++
                    }
                    @Suppress("UNCHECKED_CAST")
                    acc as V
                }

                else -> throw IllegalArgumentException("Unsupported dtype for matmul: ${a.dtype}")
            }
        }
        return CpuTensor(outData, this, a.dtype)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-transpose")
    override fun <T : DType> transpose(tensor: Tensor<T, V>): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-conv2d")
    override fun <T : DType> conv2d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        dilation: Pair<Int, Int>,
        groups: Int
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-maxpool2d")
    override fun <T : DType> maxPool2d(
        input: Tensor<T, V>,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-reshape")
    override fun <T : DType> reshape(
        tensor: Tensor<T, V>,
        newShape: Shape
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-flatten")
    override fun <T : DType> flatten(
        tensor: Tensor<T, V>,
        startDim: Int,
        endDim: Int
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-concat")
    override fun <T : DType> concat(
        tensors: List<Tensor<T, V>>,
        dim: Int
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-split")
    override fun <T : DType> split(
        tensor: Tensor<T, V>,
        splitSize: Int,
        dim: Int
    ): List<Tensor<T, V>> {
        TODO("Not yet implemented")
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-squeeze")
    override fun <T : DType> squeeze(
        tensor: Tensor<T, V>,
        dim: Int?
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-unsqueeze")
    override fun <T : DType> unsqueeze(
        tensor: Tensor<T, V>,
        dim: Int
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-relu")
    override fun <T : DType> relu(tensor: Tensor<T, V>): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-softmax")
    override fun <T : DType> softmax(
        tensor: Tensor<T, V>,
        dim: Int
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-sigmoid")
    override fun <T : DType> sigmoid(tensor: Tensor<T, V>): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-silu")
    override fun <T : DType> silu(tensor: Tensor<T, V>): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-gelu")
    override fun <T : DType> gelu(tensor: Tensor<T, V>): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-sum")
    override fun <T : DType> sum(
        tensor: Tensor<T, V>,
        dim: Int?
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-mean")
    override fun <T : DType> mean(
        tensor: Tensor<T, V>,
        dim: Int?
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-variance")
    override fun <T : DType> variance(
        tensor: Tensor<T, V>,
        dim: Int?
    ): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-sqrt")
    override fun <T : DType> sqrt(tensor: Tensor<T, V>): Tensor<T, V> {
        TODO("Not yet implemented")
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-convert")
    override fun <TFrom : DType, TTo : DType> convert(
        tensor: Tensor<TFrom, V>,
        targetType: TTo
    ): Tensor<TTo, V> {
        TODO("Not yet implemented")
    }

}