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

import sk.ai.net.safetensors.WeightLoader

class HTTPSafeTensorLoader(
    modelRoot: Path,
    owner: String?,
    modelName: String?,
    modelDType: DType?,
    branch: Optional<String?>?,
    authToken: Optional<String?>?
) : WeightLoader {
    private val modelRoot: Path
    private val indexFile: String?
    private val modelName: String
    private val branch: Optional<String?>?
    private val authToken: Optional<String?>?
    private val index: SafeTensorIndex
    private val layerFiles: Map<String?, Pair<RandomAccessFile?, AbstractTensor?>?>
    private val dynamicTensorInfoMap: Map<String?, TensorInfo?>
    private val tensorFileOffsets: Map<String?, Integer?>
    private val modelDType: DType?

    /**
     * Used for distributed inference
     *
     * Dynamically fetches weights from a remote server based on the distributed context
     *
     * @param modelRoot
     * @param owner
     * @param modelName
     * @param branch
     * @param authToken
     * @throws JsonProcessingException
     */
    init {
        this.modelRoot = modelRoot
        this.modelName = owner.toString() + "/" + modelName
        this.branch = branch
        this.indexFile = String.format("%s/%s", modelRoot, SafeTensorIndex.MODEL_INDEX_JSON)
        this.authToken = authToken

        // Check we have the index file
        if (!File(indexFile).exists()) {
            this.index =
                SafeTensorIndex(Collections.emptyMap(), Map.of("model-file", SafeTensorIndex.SINGLE_MODEL_NAME))
        } else {
            try {
                this.index = JsonSupport.om.readValue(File(indexFile), SafeTensorIndex::class.java)
            } catch (e: IOException) {
                throw RuntimeException(e)
            }
        }

        this.layerFiles = HashMap()
        this.dynamicTensorInfoMap = HashMap()
        this.tensorFileOffsets = HashMap()
        this.modelDType = modelDType
    }

    @Override
    fun metadata(): Map<String?, String?> {
        return index.metadata()
    }

    @Override
    fun tensorInfoMap(): Map<String?, TensorInfo?> {
        return dynamicTensorInfoMap
    }

    @Override
    fun load(name: String?, dctx: DistributedContext?, sparseRows: Boolean, sparseColumns: Boolean): AbstractTensor? {
        Preconditions.checkArgument(!sparseColumns || !sparseRows, "Cannot have both sparse rows and columns")
        Preconditions.checkArgument(
            index.weightFileMap.containsKey(name) || index.weightFileMap.size() === 1,
            "Unknown weight: " + name
        )

        // Check if we already have the layer loaded
        if (layerFiles.containsKey(name)) {
            return layerFiles.get(name).right()
        }

        try {
            val info: TensorInfo = maybeLoadTensorInfo(name)

            val offsets: Pair<TensorShape?, Pair<Long?, Long?>?> = Weights.getLoadOffsets(info, dctx, sparseRows)

            val headerOffset: Integer = tensorFileOffsets.get(name)

            assert(headerOffset != null && headerOffset > 0) { "Failed to find header offset for: " + name }
            val weightFile = index.weightFileMap.getOrDefault(name, SafeTensorIndex.SINGLE_MODEL_NAME)

            val shape: TensorShape? = offsets.left
            val positionOffset: Long = offsets.right.left + headerOffset
            val positionLimit: Long = offsets.right.right + headerOffset
            val length = positionLimit - positionOffset

            if (length > Integer.MAX_VALUE) {
                // Make a segmented tensor
                assert(info.shape.length === 2) { "Only 2D tensors supported" }

                val tensors: List<AbstractTensor?> = ArrayList()
                val bytesPerColumn: Int = info.dType.size() * info.shape[1]
                var offset = positionOffset
                // Chunk size needs to be a multiple of the column size
                val chunkSize: Long = Integer.MAX_VALUE - (Integer.MAX_VALUE % bytesPerColumn)
                var chunkNum = 0
                while (offset < positionLimit) {
                    val chunkEnd: Long = Math.min(offset + chunkSize, positionLimit)
                    val numRowsInChunk: Int = Ints.checkedCast((chunkEnd - offset) / bytesPerColumn)
                    val chunkShape: TensorShape? = TensorShape.of(numRowsInChunk, info.shape[1])
                    tensors.add(
                        downloadAndLoadTensor(
                            name.toString() + ".part." + chunkNum++,
                            weightFile,
                            info,
                            chunkShape,
                            offset,
                            chunkEnd,
                            dctx,
                            sparseRows,
                            sparseColumns
                        )
                    )
                    offset = chunkEnd
                }

                val wrapped: AbstractTensor? = SegmentedTensor.wrap(tensors)
                layerFiles.put(name, Pair.of(null, wrapped))

                return wrapped
            } else {
                return downloadAndLoadTensor(
                    name,
                    weightFile,
                    info,
                    shape,
                    positionOffset,
                    positionLimit,
                    dctx,
                    sparseRows,
                    sparseColumns
                )
            }
        } catch (e: IOException) {
            throw RuntimeException(e)
        }
    }

    @Throws(IOException::class)
    private fun downloadAndLoadTensor(
        name: String?,
        weightFile: String?,
        info: TensorInfo,
        shape: TensorShape?,
        positionOffset: Long,
        positionLimit: Long,
        dctx: DistributedContext?,
        sparseRows: Boolean,
        sparseColumns: Boolean
    ): AbstractTensor? {
        val weightPath: Path =
            modelRoot.resolve(weightFile.toString() + ".part." + positionOffset + "_" + positionLimit)

        if (!weightPath.toFile().exists()) {
            com.github.tjake.jlama.safetensors.HTTPSafeTensorLoader.Companion.logger.info(
                "Downloading file: {} for {} {}MB",
                weightPath,
                name,
                (positionLimit - positionOffset) / 1024 / 1024
            )
            HttpSupport.downloadFile(
                modelName,
                weightFile,
                branch,
                authToken,
                Optional.of(Pair.of(positionOffset, positionLimit)),
                weightPath,
                Optional.empty()
            )
        }

        val length: Int = Ints.checkedCast(positionLimit - positionOffset)

        val raf: RandomAccessFile = RandomAccessFile(weightPath.toFile(), "r")
        val buf: ByteBuffer? = raf.getChannel()
            .map(FileChannel.MapMode.READ_ONLY, 0, raf.length())
            .duplicate()
            .order(ByteOrder.LITTLE_ENDIAN)
            .position(0)
            .limit(length)

        if (raf.length() < length) {
            throw RuntimeException(
                "Failed to download the correct number of bytes: " + raf.length() + " != " + length + " for " + weightPath
            )
        }

        com.github.tjake.jlama.safetensors.HTTPSafeTensorLoader.Companion.logger.debug(
            "Loading tensor: {} from {} with offsets: {} {}",
            name,
            weightPath,
            positionOffset,
            positionLimit
        )

        val tensor: AbstractTensor? = Weights.loadTensorFromBuffer(
            name,
            info.dType,
            modelDType,
            shape,
            buf,
            sparseRows,
            sparseColumns,
            dctx,
            this
        )

        layerFiles.put(name, Pair.of(raf, tensor))

        return tensor
    }

    @Throws(IOException::class)
    private fun maybeLoadTensorInfo(name: String?): TensorInfo {
        if (dynamicTensorInfoMap.containsKey(name)) {
            return dynamicTensorInfoMap.get(name)
        }

        val weightFile = index.weightFileMap.getOrDefault(name, SafeTensorIndex.SINGLE_MODEL_NAME)

        val headerFile: Path = modelRoot.resolve(weightFile.toString() + ".header")

        if (!Files.exists(headerFile)) {
            // Download the first 1MB of the file to get the tensor info
            HttpSupport.downloadFile(
                modelName,
                weightFile,
                branch,
                authToken,
                Optional.of(Pair.of(0L, 1L shl 20)),
                headerFile,
                Optional.empty()
            )
        }

        RandomAccessFile(headerFile.toFile(), "r").use { raf ->
            val header: ByteBuffer =
                raf.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, Math.min(1 shl 20, raf.length()))
            val info: Map<String?, TensorInfo?> = SafeTensorSupport.readTensorInfoMap(header, Optional.empty())
            val endOfHeaderPosition: Int = header.position()
            for (e in info.entrySet()) {
                dynamicTensorInfoMap.put(e.getKey(), e.getValue())
                tensorFileOffsets.put(e.getKey(), endOfHeaderPosition)
            }
        }
        assert(dynamicTensorInfoMap.containsKey(name)) { "Failed to load tensor info for: " + name }
        return dynamicTensorInfoMap.get(name)
    }

    @Override
    fun getModelDType(): DType? {
        return modelDType
    }

    @Override
    fun close() {
        for (pair in layerFiles.values()) {
            try {
                if (pair.left() != null) pair.left().close()
            } catch (e: IOException) {
                throw RuntimeException(e)
            }
        }

        layerFiles.clear()
        dynamicTensorInfoMap.clear()
    }

    companion object {
        private val logger: Logger =
            LoggerFactory.getLogger(com.github.tjake.jlama.safetensors.HTTPSafeTensorLoader::class.java)
    }
}
