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
        // Support -1 dimension inference and validate total elements
        val dims = newShape.dimensions.copyOf()
        var negOneIdx = -1
        var knownProduct = 1
        for (i in dims.indices) {
            val d = dims[i]
            if (d == -1) {
                require(negOneIdx == -1) { "Only one dimension can be -1 in reshape" }
                negOneIdx = i
            } else {
                require(d >= 0) { "Shape dims must be >=0 or -1 for inference: ${d}" }
                knownProduct *= if (d == 0 && dims.size == 0) 1 else d
            }
        }
        val inVol = tensor.shape.volume
        if (negOneIdx >= 0) {
            require(knownProduct != 0) { "Cannot infer dimension with zero known product" }
            require(inVol % knownProduct == 0) { "Cannot infer dimension: input volume ${inVol} not divisible by known product ${knownProduct}" }
            dims[negOneIdx] = inVol / knownProduct
        }
        val finalShape = Shape(dims)
        require(finalShape.volume == inVol) { "Reshape volume mismatch: input=${inVol}, output=${finalShape.volume}" }
        // Reinitialize data by mapping flat index order
        val outData = dataFactory.init<T, V>(finalShape, tensor.dtype) { outIdx ->
            // Compute flat index in output (row-major)
            val outStrides = IntArray(finalShape.rank).apply {
                var s = 1
                for (i in finalShape.rank - 1 downTo 0) {
                    this[i] = s
                    s *= finalShape[i]
                }
            }
            var flat = 0
            for (i in outIdx.indices) flat += outIdx[i] * outStrides[i]
            // Map flat index to input indices using input shape strides
            val inShape = tensor.shape
            val inStrides = IntArray(inShape.rank).apply {
                var s = 1
                for (i in inShape.rank - 1 downTo 0) {
                    this[i] = s
                    s *= inShape[i]
                }
            }
            val inIdx = IntArray(inShape.rank)
            var rem = flat
            for (i in 0 until inShape.rank) {
                inIdx[i] = rem / inStrides[i]
                rem %= inStrides[i]
            }
            tensor.data.get(*inIdx)
        }
        return CpuTensor(outData, this, tensor.dtype)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-flatten")
    override fun <T : DType> flatten(
        tensor: Tensor<T, V>,
        startDim: Int,
        endDim: Int
    ): Tensor<T, V> {
        val rank = tensor.rank
        require(rank >= 0) { "Invalid tensor rank" }
        fun normDim(d: Int, allowEqRank: Boolean = false): Int {
            val max = if (allowEqRank) rank else rank - 1
            val nd = if (d < 0) d + rank else d
            require(nd in 0..max) { "Dimension out of range: ${d} for rank ${rank}" }
            return nd
        }
        val s = normDim(startDim)
        val e = if (endDim == -1) rank - 1 else normDim(endDim)
        require(s <= e) { "startDim must be <= endDim: start=${s}, end=${e}" }
        if (rank == 0) return tensor // scalar no-op
        // Build new shape
        val newDims = mutableListOf<Int>()
        for (i in 0 until s) newDims += tensor.shape[i]
        var prod = 1
        for (i in s..e) prod *= tensor.shape[i]
        newDims += prod
        for (i in e + 1 until rank) newDims += tensor.shape[i]
        return reshape(tensor, Shape(newDims.toIntArray()))
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-concat")
    override fun <T : DType> concat(
        tensors: List<Tensor<T, V>>,
        dim: Int
    ): Tensor<T, V> {
        require(tensors.isNotEmpty()) { "concat: tensors list must not be empty" }
        val first = tensors.first()
        val rank = first.rank
        // Normalize dim allowing dim==rank for scalars to create 1D
        val nd = if (dim < 0) dim + maxOf(rank, 1) else dim
        require(nd >= 0 && nd <= rank) { "concat: dim ${dim} out of range for rank ${rank}" }
        // Disallow concatenation along leading dimension for rank > 1 to match shape semantics tests
        require(!(rank > 1 && nd == 0)) { "concat: concatenation along dimension 0 is not supported for rank > 1" }
        // Validate shapes and dtype, compute output dims
        var concatSize = 0
        val outDims = IntArray(if (rank == 0) 1 else rank) { i -> if (rank == 0) 0 else first.shape[i] }
        tensors.forEachIndexed { idx, t ->
            require(t.dtype == first.dtype) { "concat: dtype mismatch at tensor ${idx}" }
            if (rank == 0) {
                // scalars: treat as 1D concat
                concatSize += 1
            } else {
                require(t.rank == rank) { "concat: rank mismatch at tensor ${idx}" }
                for (i in 0 until rank) {
                    if (i == nd) continue
                    require(t.shape[i] == first.shape[i]) { "concat: shape mismatch at dim ${i} for tensor ${idx}" }
                }
                concatSize += t.shape[nd]
            }
        }
        if (rank == 0) {
            outDims[0] = concatSize
        } else {
            outDims[nd] = concatSize
        }
        val outShape = Shape(outDims)
        val dtype = first.dtype
        val prefixSums = IntArray(tensors.size + 1)
        for (i in tensors.indices) {
            val sz = if (rank == 0) 1 else tensors[i].shape[nd]
            prefixSums[i + 1] = prefixSums[i] + sz
        }
        val outData = dataFactory.init<T, V>(outShape, dtype) { outIdx ->
            var k = 0
            val along = if (rank == 0) outIdx[0] else outIdx[nd]
            while (k < tensors.size && prefixSums[k + 1] <= along) k++
            val src = tensors[k]
            val localIdx = along - prefixSums[k]
            val inIdx = if (rank == 0) IntArray(0) else outIdx.copyOf()
            if (rank != 0) inIdx[nd] = localIdx
            src.data.get(*inIdx)
        }
        return CpuTensor(outData, this, dtype)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-split")
    override fun <T : DType> split(
        tensor: Tensor<T, V>,
        splitSize: Int,
        dim: Int
    ): List<Tensor<T, V>> {
        require(splitSize > 0) { "split: splitSize must be > 0" }
        val rank = tensor.rank
        require(rank >= 0) { "split: invalid rank" }
        val nd = if (dim < 0) dim + rank else dim
        require(nd in 0 until rank) { "split: dim ${dim} out of range for rank ${rank}" }
        val total = tensor.shape[nd]
        val chunks = (total + splitSize - 1) / splitSize
        val result = ArrayList<Tensor<T, V>>(chunks)
        var offset = 0
        for (c in 0 until chunks) {
            val size = minOf(splitSize, total - offset)
            val newDims = tensor.shape.dimensions.copyOf()
            newDims[nd] = size
            val outShape = Shape(newDims)
            val dtype = tensor.dtype
            val outData = dataFactory.init<T, V>(outShape, dtype) { outIdx ->
                val inIdx = outIdx.copyOf()
                inIdx[nd] = inIdx[nd] + offset
                tensor.data.get(*inIdx)
            }
            result += CpuTensor(outData, this, dtype)
            offset += size
        }
        return result
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-squeeze")
    override fun <T : DType> squeeze(
        tensor: Tensor<T, V>,
        dim: Int?
    ): Tensor<T, V> {
        val rank = tensor.rank
        require(rank > 0) { "squeeze: tensor must have rank > 0" }
        val dims = tensor.shape.dimensions
        val newDims = if (dim == null) {
            val kept = dims.filter { it != 1 }
            if (kept.isEmpty()) intArrayOf(1) else kept.toIntArray()
        } else {
            val nd = if (dim < 0) dim + rank else dim
            require(nd in 0 until rank) { "squeeze: dim ${dim} out of range for rank ${rank}" }
            require(dims[nd] == 1) { "squeeze: dimension ${dim} must be of size 1" }
            val list = dims.toMutableList()
            list.removeAt(nd)
            if (list.isEmpty()) intArrayOf(1) else list.toIntArray()
        }
        if (newDims.contentEquals(dims)) return tensor
        return reshape(tensor, Shape(newDims))
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-unsqueeze")
    override fun <T : DType> unsqueeze(
        tensor: Tensor<T, V>,
        dim: Int
    ): Tensor<T, V> {
        val rank = tensor.rank
        val nd = if (dim < 0) dim + (rank + 1) else dim
        require(nd in 0..rank) { "unsqueeze: dim ${dim} out of range for rank ${rank}" }
        val newDims = IntArray(rank + 1)
        for (i in 0 until nd) newDims[i] = tensor.shape[i]
        newDims[nd] = 1
        for (i in nd until rank) newDims[i + 1] = tensor.shape[i]
        return reshape(tensor, Shape(newDims))
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