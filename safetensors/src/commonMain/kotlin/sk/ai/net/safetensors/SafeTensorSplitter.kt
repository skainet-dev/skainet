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

import com.github.tjake.jlama.tensor.AbstractTensor
import sk.ai.net.safetensors.WeightLoader

/** Helper class to split a large model into pieces  */
object SafeTensorSplitter {
    // Limit chunk size to 20G
    var MAX_CHUNK_SIZE: Long = 20L shl 30

    fun getChunkFile(info: TensorInfo, fileSize: Long): String {
        // Map Tensor to a chunk based on its location in the model
        val fileChunk: Long =
            Math.floorDiv(info.dataOffsets[1], com.github.tjake.jlama.safetensors.SafeTensorSplitter.MAX_CHUNK_SIZE)
        val totalChunks: Long =
            Math.floorDiv(fileSize, com.github.tjake.jlama.safetensors.SafeTensorSplitter.MAX_CHUNK_SIZE)
        return String.format("model-%05d-of-%05d.safetensor", fileChunk, totalChunks)
    }

    fun main(args: Array<String?>) {
        require(args.size != 0) { "Missing model name" }

        val modelDir = args[0]

        require(File(modelDir).isDirectory()) { "Not a directory" }

        require(!Paths.get(modelDir, SafeTensorIndex.MODEL_INDEX_JSON).toFile().exists()) { "Already split" }

        require(Paths.get(modelDir, SafeTensorIndex.SINGLE_MODEL_NAME).toFile().exists()) { "Missing model file" }

        val wl: WeightLoader = SafeTensorSupport.loadWeights(File(modelDir))

        try {
            val info: Map<String?, TensorInfo?> = wl.tensorInfoMap()

            // First split the metadata into N chunks and adjust the offsets
            val tensorIndex: Map<String?, String?> = LinkedHashMap()
            val chunkFiles: Map<String?, Pair<RandomAccessFile?, FileChannel?>?> = HashMap()

            val tensorsInChunk: Map<String?, Map<String?, TensorInfo?>?> = LinkedHashMap()

            for (entry in info.entrySet()) {
                val tensorInfo: TensorInfo = entry.getValue()
                val name: String? = entry.getKey()

                val chunkName: String = com.github.tjake.jlama.safetensors.SafeTensorSplitter.getChunkFile(
                    tensorInfo,
                    File(modelDir, SafeTensorIndex.SINGLE_MODEL_NAME).length()
                )
                tensorIndex.put(name, chunkName)

                val chunkFile: Pair<RandomAccessFile?, FileChannel?> = chunkFiles.computeIfAbsent(chunkName, { n ->
                    try {
                        val tmp: File = File.createTempFile("jlama", "chunk")
                        tmp.deleteOnExit()
                        val r: RandomAccessFile = RandomAccessFile(tmp, "rw")
                        val ch: FileChannel? = r.getChannel()

                        return@computeIfAbsent Pair.of(r, ch)
                    } catch (e: IOException) {
                        throw RuntimeException(e)
                    }
                })

                val t: AbstractTensor = wl.load(name)
                val ch: FileChannel? = chunkFile.right
                val newInfo: TensorInfo = t.save(ch)
                System.out.println(
                    "Wrote " + name + " to " + chunkName + " at " + newInfo.dataOffsets[0] + " to " + newInfo.dataOffsets[1]
                )

                val tensors: Map<String?, TensorInfo?> =
                    tensorsInChunk.computeIfAbsent(chunkName, { n -> LinkedHashMap() })
                tensors.put(name, newInfo)
            }

            // Now We have the data im place data, write the real file
            for (entry in chunkFiles.entrySet()) {
                val chunkName: String? = entry.getKey()
                val chunkFile: Pair<RandomAccessFile?, FileChannel?> = entry.getValue()

                val ch: FileChannel = chunkFile.left.getChannel()
                val chunkTensors: Map<String?, TensorInfo?> = tensorsInChunk.get(chunkName)

                val header: ByteArray = om.writeValueAsBytes(chunkTensors)
                System.out.println("Writing " + chunkName + " with " + chunkTensors.size() + " tensors")
                // System.out.println(new String(header));
                val hsize = ByteArray(Long.BYTES)
                ByteBuffer.wrap(hsize).order(ByteOrder.LITTLE_ENDIAN).putLong(header.size)

                RandomAccessFile(Paths.get(modelDir, chunkName).toFile(), "rw").use { raf ->
                    raf.write(hsize)
                    raf.write(header)
                    raf.seek(raf.length())
                    System.out.println("Writing " + ch.size() + " bytes of data from " + raf.getChannel().position())
                    ch.transferTo(0, ch.size(), raf.getChannel())
                }
            }

            RandomAccessFile(Paths.get(modelDir, SafeTensorIndex.MODEL_INDEX_JSON).toFile(), "rw").use { raf ->
                raf.write(om.writeValueAsBytes(Map.of("metadata", HashMap(), "weight_map", tensorIndex)))
            }
            // Clean up
            for (p in chunkFiles.values()) {
                p.left.close()
            }
        } catch (e: IOException) {
            throw RuntimeException(e)
        }
    }
}
