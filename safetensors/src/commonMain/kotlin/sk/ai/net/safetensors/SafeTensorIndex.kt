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

import com.fasterxml.jackson.annotation.JsonCreator
import sk.ai.net.safetensors.WeightLoader

class SafeTensorIndex @JsonCreator internal constructor(
    @JsonProperty("metadata") metadata: Map<String?, String?>?,
    @JsonProperty("weight_map") weightFileMap: Map<String?, String?>?
) : WeightLoader, AutoCloseable {
    private val metadata: Map<String?, String?>?

    val allTensorInfoMap: Map<String?, TensorInfo?> = HashMap()

    // Map from weight name to file name (this is what's in the JSON file)
    val weightFileMap: Map<String?, String?>

    // Map from weight name to Weights data
    private val weightMap: Map<String?, Weights?> = HashMap()

    // Map from file name to RandomAccessFile
    private val fileMap: Map<String?, RandomAccessFile?> = HashMap()

    /**
     * Group tensors into splits that can be mmaped together.
     * Since mmap limitation is integer max_value length.
     *
     * This also adjusts (inplace) the tensor offsets to be relative to the start of the split.
     *
     */
    private fun computeMmapSplits(
        tensorInfoMap: Map<String?, TensorInfo?>,
        fileLength: Long
    ): Map<List<Long?>?, List<String?>?> {
        val splits: Map<List<Long?>?, List<String?>?> = HashMap()
        var lastSplitOffset: Long = 0
        val tensorsInFile: Int = tensorInfoMap.size()
        var tensorsSplit = 0
        val tensors: List<String?> = ArrayList()

        val it: Iterator<Map.Entry<String?, TensorInfo?>?> = ArrayList(tensorInfoMap.entrySet()).iterator()
        var next: Map.Entry<String?, TensorInfo?>? = null
        while (tensorsSplit < tensorsInFile && (it.hasNext() || next != null)) {
            tensors.clear()
            val limit: Long = lastSplitOffset + Integer.MAX_VALUE
            var startOffset = fileLength
            var endOffset: Long = 0

            while (it.hasNext() || next != null) {
                next = if (next == null) it.next() else next
                val info: TensorInfo = next.getValue()
                com.github.tjake.jlama.safetensors.SafeTensorIndex.Companion.logger.debug(
                    "Tensor {} {} {} limit {}",
                    next.getKey(),
                    info.dataOffsets[0],
                    info.dataOffsets[1],
                    limit
                )
                if (info.dataOffsets[1] < limit) {
                    tensors.add(next.getKey())
                    tensorsSplit++

                    if (info.dataOffsets[1] > endOffset) endOffset = info.dataOffsets[1]
                    if (info.dataOffsets[0] < startOffset) startOffset = info.dataOffsets[0]

                    // Adjust the offset to be relative to the start of the split
                    info.dataOffsets[0] -= lastSplitOffset
                    info.dataOffsets[1] -= lastSplitOffset

                    com.github.tjake.jlama.safetensors.SafeTensorIndex.Companion.logger.debug(
                        "Adding tensor {} to split {}-{}",
                        next.getKey(),
                        info.dataOffsets[0],
                        info.dataOffsets[1]
                    )

                    // Used so fetch the tensor from the mmap
                    next = null
                } else {
                    // Split large tensors up (they will be reassembled in the Weights class)
                    if (tensors.size() === 0) {
                        val bytesPerColumn: Int = info.dType.size() * info.shape[1]

                        // This tensor is too large to fit in a single split
                        // We'll split it up into smaller chunks
                        if (info.dataOffsets[1] > endOffset) endOffset = info.dataOffsets[1]
                        if (info.dataOffsets[0] < startOffset) startOffset = info.dataOffsets[0]

                        // Adjust the offset to be relative to the start of the split
                        info.dataOffsets[0] -= lastSplitOffset
                        info.dataOffsets[1] -= lastSplitOffset

                        var offset: Long = info.dataOffsets[0]!!
                        var length: Long = info.dataOffsets[1] - offset

                        // Chunk size needs to be a multiple of the column size
                        val chunkSize: Long = Integer.MAX_VALUE - (Integer.MAX_VALUE % bytesPerColumn)
                        var offsetAdded: Long = 0
                        var chunk = 0
                        var added = false
                        while (length > 0) {
                            val chunkEnd: Long = Math.min(offset + chunkSize, endOffset)
                            val chunkName: String = next.getKey() + "-part-" + chunk++
                            com.github.tjake.jlama.safetensors.SafeTensorIndex.Companion.logger.debug(
                                "Adding chunk {} to split {}-{} {}",
                                chunkName,
                                offset,
                                chunkEnd,
                                Ints.checkedCast(chunkEnd - offset)
                            )
                            splits.put(List.of(offset, chunkEnd), List.of(chunkName))

                            // Add TensorInfo for the chunk
                            assert(info.shape.length === 2) { "Only 2D tensors supported" }
                            val numRowsInChunk: Int = Ints.checkedCast((chunkEnd - offset) / bytesPerColumn)

                            // This tensorInfo is relative to the split which we know is at least the mmap limit
                            // We track the offsetAdded so we can make the offset relative to the current split
                            val chunkInfo: TensorInfo = TensorInfo(
                                info.dType,
                                longArrayOf(numRowsInChunk.toLong(), info.shape[1]),
                                longArrayOf(offset - offsetAdded, chunkEnd - offsetAdded)
                            )
                            tensorInfoMap.put(chunkName, chunkInfo)
                            added = true
                            offsetAdded += chunkEnd - offset

                            offset = chunkEnd
                            length -= chunkSize
                        }

                        if (added) {
                            tensorsSplit++
                            next = null
                        }
                    }

                    break
                }
            }

            assert(tensorsSplit > 0) { "No tensors in split" }
            com.github.tjake.jlama.safetensors.SafeTensorIndex.Companion.logger.debug(
                "Adding split {}-{} with {} tensors of {}",
                startOffset,
                endOffset,
                tensors.size(),
                tensorsSplit
            )

            // Add any sections that were split
            if (!tensors.isEmpty()) splits.put(List.of(startOffset, endOffset), ArrayList(tensors))

            if (endOffset > lastSplitOffset) lastSplitOffset = endOffset
        }

        assert(tensorsInFile == tensorsSplit) { "Not all tensors were split: " + tensorsSplit + " != " + tensorsInFile }
        return splits
    }

    init {
        this.metadata = ImmutableMap.copyOf(metadata)
        this.weightFileMap = ImmutableMap.copyOf(weightFileMap)
    }

    @Override
    fun metadata(): Map<String?, String?>? {
        return metadata
    }

    @Override
    fun tensorInfoMap(): Map<String?, TensorInfo?> {
        return allTensorInfoMap
    }

    @Override
    fun load(name: String?, dctx: DistributedContext?, sparseRows: Boolean, sparseColumns: Boolean): AbstractTensor {
        val w: Weights? = weightMap.get(name)
        if (w == null) {
            // Maybe assemble the tensor from segments
            val segments: List<AbstractTensor?> = ArrayList()
            var idx = 0
            while (true) {
                val segmentName = name.toString() + "-part-" + idx++
                if (!weightMap.containsKey(segmentName)) break
                segments.add(weightMap.get(segmentName).load(segmentName, dctx, sparseRows, sparseColumns))
            }

            if (segments.size() > 0) {
                return SegmentedTensor.wrap(segments)
            }

            throw NoSuchElementException(name)
        }

        return w.load(name, dctx, sparseRows, sparseColumns)
    }

    @get:Override
    val modelDType: DType
        get() =// FIXME: This assumes all weights have the same dtype
            weightMap.values().iterator().next().getModelDType()

    @Override
    @Throws(Exception::class)
    fun close() {
        weightMap.clear()
        fileMap.forEach({ k, v ->
            try {
                v.close()
            } catch (e: IOException) {
                // Close quietly
            }
        })
        fileMap.clear()
        allTensorInfoMap.clear()
    }

    companion object {
        private val logger: Logger =
            LoggerFactory.getLogger(com.github.tjake.jlama.safetensors.SafeTensorIndex::class.java)
        private val om: ObjectMapper = ObjectMapper()

        val SINGLE_MODEL_NAME: String = "model.safetensors"
        val MODEL_INDEX_JSON: String = "model.safetensors.index.json"

        fun loadWithWeights(modelRoot: Path): SafeTensorIndex {
            try {
                val indexFile: File? = Paths.get(
                    modelRoot.toString(),
                    com.github.tjake.jlama.safetensors.SafeTensorIndex.Companion.MODEL_INDEX_JSON
                ).toFile()

                val index: SafeTensorIndex = com.github.tjake.jlama.safetensors.SafeTensorIndex.Companion.om.readValue(
                    indexFile,
                    com.github.tjake.jlama.safetensors.SafeTensorIndex::class.java
                )
                com.github.tjake.jlama.safetensors.SafeTensorIndex.Companion.loadWeights(index, modelRoot)

                return index
            } catch (e: IOException) {
                throw RuntimeException(e)
            }
        }

        fun loadSingleFile(modelRoot: Path, modelFile: String?): SafeTensorIndex {
            try {
                val index: SafeTensorIndex = com.github.tjake.jlama.safetensors.SafeTensorIndex(
                    Collections.emptyMap(),
                    Map.of("model-file", modelFile)
                )
                com.github.tjake.jlama.safetensors.SafeTensorIndex.Companion.loadWeights(index, modelRoot)

                return index
            } catch (e: IOException) {
                throw RuntimeException(e)
            }
        }

        @Throws(IOException::class)
        fun loadWeights(index: SafeTensorIndex, modelRoot: Path) {
            for (e in index.weightFileMap.entrySet()) {
                // Only load the file if it's not already loaded
                if (!index.fileMap.containsKey(e.getValue())) {
                    val raf: RandomAccessFile =
                        RandomAccessFile(Paths.get(modelRoot.toString(), e.getValue()).toFile(), "r")
                    index.fileMap.put(e.getValue(), raf)

                    // Read the first 1MB of the file to get the TensorInfo
                    val header: ByteBuffer =
                        raf.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, Math.min(1 shl 20, raf.length()))

                    val metadata: Map<String?, String?> = HashMap()
                    val tensorInfoMap: Map<String?, TensorInfo?> =
                        SafeTensorSupport.readTensorInfoMap(header, Optional.of(metadata))
                    index.allTensorInfoMap.putAll(tensorInfoMap)
                    val endOfHeaderPosition: Int = header.position()

                    val splits = index.computeMmapSplits(tensorInfoMap, raf.length())
                    for (split in splits.entrySet()) {
                        val offset: Long = split.getKey().get(0)!!
                        val length: Long = split.getKey().get(1)!!
                        val tensors: List<String?> = split.getValue()
                        val lengthInt: Int = Ints.checkedCast(length - offset)

                        val buf: ByteBuffer? =
                            raf.getChannel().map(FileChannel.MapMode.READ_ONLY, endOfHeaderPosition + offset, lengthInt)

                        val mmapTensorInfoMap: Map<String?, TensorInfo?>? = tensorInfoMap.entrySet()
                            .stream()
                            .filter({ x -> tensors.contains(x.getKey()) })
                            .collect(ImmutableMap.toImmutableMap(Map.Entry::getKey, Map.Entry::getValue))

                        val mmapWeights: Weights = Weights(metadata, mmapTensorInfoMap, buf, Optional.of(index))
                        for (tensor in tensors) {
                            index.weightMap.put(tensor, mmapWeights)
                        }
                    }
                }
            }
        }
    }
}
