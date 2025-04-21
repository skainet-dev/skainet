/*
 * Copyright 2024 T Jake Luciani
 *
 * The Jlama Project licenses this file to you under the Apache License,
 * version 2.0 (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.github.tjake.jlama.safetensors

import com.github.tjake.jlama.model.DistributedContext
import sk.ai.net.safetensors.WeightLoader

class Weights internal constructor(
    metadata: Map<String?, String?>?,
    tensorInfoMap: Map<String?, TensorInfo?>,
    bytes: ByteBuffer,
    parent: Optional<WeightLoader?>
) : WeightLoader {
    private val metadata: Map<String?, String?>?
    private val tensorInfoMap: Map<String?, TensorInfo?>
    private val bytes: ByteBuffer
    private val majorityDType: DType?
    private val parent: Optional<WeightLoader?>

    init {
        this.metadata = ImmutableMap.copyOf(metadata)
        this.tensorInfoMap = ImmutableMap.copyOf(tensorInfoMap)
        this.bytes = bytes.duplicate()
        this.majorityDType = com.github.tjake.jlama.safetensors.Weights.Companion.findDType(tensorInfoMap)
        this.parent = parent
    }

    @Override
    fun metadata(): Map<String?, String?>? {
        return metadata
    }

    @Override
    fun tensorInfoMap(): Map<String?, TensorInfo?> {
        return tensorInfoMap
    }

    @Override
    fun load(name: String?, dctx: DistributedContext?, sparseRows: Boolean, sparseColumns: Boolean): AbstractTensor? {
        val info: TensorInfo = tensorInfoMap.get(name)
        if (info == null) throw NoSuchElementException(name.toString() + " not found in weights")

        if (info.shape.length < 1) throw RuntimeException("Invalid shape dimensions " + info.shape.length + " encountered for " + name)

        if (dctx != null && info.shape.length !== 2) {
            throw RuntimeException("Invalid shape dimensions " + info.shape.length + " encountered for " + name + " with offset")
        }

        val offsets: Pair<TensorShape?, Pair<Long?, Long?>?> =
            com.github.tjake.jlama.safetensors.Weights.Companion.getLoadOffsets(info, dctx, sparseRows)

        val b: ByteBuffer = bytes.duplicate()
            .order(ByteOrder.LITTLE_ENDIAN)
            .position(Ints.checkedCast(offsets.right.left))
            .limit(Ints.checkedCast(offsets.right.right))

        return com.github.tjake.jlama.safetensors.Weights.Companion.loadTensorFromBuffer(
            name,
            info.dType,
            majorityDType,
            offsets.left,
            b,
            sparseRows,
            sparseColumns,
            dctx,
            parent.orElse(this)
        )
    }

    @get:Override
    val modelDType: DType?
        get() = majorityDType

    @Override
    fun toString(): String? {
        return "SafeTensor{" + "metadata=" + metadata + ", tensorInfoMap=" + tensorInfoMap + ", bytes=" + bytes + '}'
    }

    @Override
    fun equals(o: Object?): Boolean {
        if (this === o) return true
        if (o == null || getClass() !== o.getClass()) return false
        val weights = o as Weights
        return Objects.equals(metadata, weights.metadata) && Objects.equals(tensorInfoMap, weights.tensorInfoMap)
    }

    @Override
    fun hashCode(): Int {
        return Objects.hash(metadata, tensorInfoMap)
    }

    @Override
    @Throws(Exception::class)
    fun close() {
    }

    companion object {
        private val logger: Logger? = LoggerFactory.getLogger(com.github.tjake.jlama.safetensors.Weights::class.java)
        fun findDType(tensorInfoMap: Map<String?, TensorInfo?>): DType? {
            val counts: EnumMap<DType?, Integer?> = EnumMap(DType::class.java)
            for (e in tensorInfoMap.entrySet()) {
                if (!e.getKey().endsWith(".qb")) counts.put(
                    e.getValue().dType,
                    counts.getOrDefault(e.getValue().dType, 0) + 1
                )
            }

            var max = 0
            var maxType: DType? = null
            for (e in counts.entrySet()) {
                if (e.getValue() > max) {
                    max = e.getValue()
                    maxType = e.getKey()
                }
            }

            // FIXME don't really support F16 atm
            return if (maxType === DType.F16) DType.F32 else maxType
        }

        fun getLoadOffsets(
            info: TensorInfo,
            dctx: DistributedContext?,
            sparseRows: Boolean
        ): Pair<TensorShape?, Pair<Long?, Long?>?> {
            var positionOffset: Long = info.dataOffsets[0]!!
            var positionLimit: Long = info.dataOffsets[1]!!
            var shape: TensorShape? = TensorShape.of(info.shape)

            // If this is a sparse tensor, we need to fetch only the section of the tensor that is needed
            if (dctx != null && sparseRows) {
                val rows: Int = info.shape[0]!!
                var columnLength: Int = info.shape[1] * info.dType.size()

                // Hack for Q4
                if (info.dType === DType.Q4) columnLength /= 2

                positionOffset = info.dataOffsets[0] + (dctx.getShardOffsetForLength(rows) * columnLength)
                positionLimit = positionOffset + (dctx.getShardLength(rows) * columnLength)
                shape = TensorShape.sparseRow(
                    info.shape,
                    Pair.of(dctx.getShardOffsetForLength(rows), dctx.getShardLength(rows))
                )
            }
            return Pair.of(shape, Pair.of(positionOffset, positionLimit))
        }

        fun loadTensorFromBuffer(
            name: String?,
            dType: DType,
            majorityDType: DType?,
            shape: TensorShape,
            b: ByteBuffer,
            sparseRows: Boolean,
            sparseColumns: Boolean,
            dctx: DistributedContext?,
            loader: WeightLoader
        ): AbstractTensor? {
            val len: Int
            val fb: FloatBuffer?
            val sb: ShortBuffer?
            val t: AbstractTensor?
            when (dType) {
                F32 -> {
                    fb = b.asFloatBuffer().slice()
                    t = FloatBufferTensor(name, fb, shape, true)
                }

                F16 ->                 // If the majority of the weights are F32 then convert to F32
                    if (majorityDType === DType.F32) {
                        len = b.remaining() / DType.F16.size()
                        val bb: ByteBuffer = ByteBuffer.allocate(len * DType.F32.size()).order(ByteOrder.LITTLE_ENDIAN)
                        var i = 0
                        while (i < len * DType.F32.size()) {
                            val s: Short = b.getShort()
                            val v: Float = Float.float16ToFloat(s)
                            bb.putFloat(i, v)
                            i += DType.F32.size()
                        }
                        t = FloatBufferTensor(bb.asFloatBuffer(), shape, true)
                    } else {
                        sb = b.asShortBuffer().slice()
                        t = Float16BufferTensor(name, sb, shape, true)
                    }

                BF16 -> {
                    sb = b.asShortBuffer().slice()
                    t = BFloat16BufferTensor(name, sb, shape, true)
                }

                Q4 -> {
                    val qb: FloatBufferTensor? = loader.load(
                        name.toString() + ".qb",
                        dctx,
                        sparseRows,
                        false /*only need sparsify once*/
                    ) as FloatBufferTensor?
                    t = Q4ByteBufferTensor(name, b.slice(), qb, shape, true)
                }

                I8 -> {
                    val qb1: FloatBufferTensor? = loader.load(
                        name.toString() + ".qb",
                        dctx,
                        sparseRows,
                        false /*only need to sparsify once*/
                    ) as FloatBufferTensor?
                    t = Q8ByteBufferTensor(name, b.slice(), qb1, shape, true)
                }

                else -> throw IllegalArgumentException("Unsupported Tensor type: " + dType.name() + " for " + name)
            }

            return if (dctx != null && sparseColumns && dctx.hasModelShard())
                t.sparsify(dctx.getShardOffsetForLength(shape.last()), dctx.getShardLength(shape.last()))
            else
                t
        }
    }
}
