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

import com.github.tjake.jlama.util.JsonSupport.om
import sk.ai.net.safetensors.WeightLoader

object SafeTensorSupport {
    private val logger: Logger = LoggerFactory.getLogger(SafeTensorSupport::class.java)
    private val metadataTypeReference: MapType? =
        om.getTypeFactory().constructMapType(Map::class.java, String::class.java, String::class.java)

    fun readTensorInfoMap(buf: ByteBuffer, saveMetadata: Optional<Map<String?, String?>?>): Map<String?, TensorInfo?>? {
        var buf: ByteBuffer = buf
        val MAX_HEADER_LENGTH = (1024 * 1024 * 1024).toLong() // 1 GB
        buf = buf.order(ByteOrder.LITTLE_ENDIAN)
        val headerLength: Long = buf.getLong()

        // headerLength is negative
        require(headerLength >= 0) { "Header length cannot be negative: " + headerLength }
        // headerLength exceeds the maximum allowed length MAX_HEADER_LENGTH
        if (headerLength > MAX_HEADER_LENGTH) {
            throw IllegalArgumentException(
                String.format(
                    "Header length %d exceeds the maximum allowed length %d.",
                    headerLength,
                    MAX_HEADER_LENGTH
                )
            )
        }

        val header = ByteArray(Ints.checkedCast(headerLength))
        buf.get(header)

        try {
            val rootNode: JsonNode = om.readTree(header)
            val fields: Iterator<Map.Entry<String?, JsonNode?>?> = rootNode.fields()
            val tensorInfoMap: Map<String?, TensorInfo?> = HashMap()
            var metadata: Map<String?, String?>? = Collections.emptyMap()

            while (fields.hasNext()) {
                val field: Map.Entry<String?, JsonNode?> = fields.next()
                if (field.getKey().equalsIgnoreCase("__metadata__")) {
                    metadata = om.treeToValue(
                        field.getValue(),
                        com.github.tjake.jlama.safetensors.SafeTensorSupport.metadataTypeReference
                    )
                } else {
                    val tensorInfo: TensorInfo? = om.treeToValue(field.getValue(), TensorInfo::class.java)
                    tensorInfoMap.put(field.getKey(), tensorInfo)
                }
            }

            // Sort by value using a lambda expression
            val sortedMap: Map<String?, TensorInfo?>? = tensorInfoMap.entrySet()
                .stream()
                .sorted(Map.Entry.comparingByValue())
                .collect(
                    Collectors.toMap(
                        Map.Entry::getKey,
                        Map.Entry::getValue,
                        { e1, e2 -> e1 },
                        { LinkedHashMap() })
                )

            val finalMetadata = metadata
            saveMetadata.ifPresent({ m -> m.putAll(finalMetadata) })

            return sortedMap
        } catch (e: IOException) {
            throw RuntimeException(e)
        }
    }

    fun readWeights(safeBuf: ByteBuffer): Weights {
        var safeBuf: ByteBuffer = safeBuf
        safeBuf = safeBuf.duplicate()
        val metadata: Map<String?, String?> = HashMap()
        val tensorInfoMap: Map<String?, TensorInfo?>? =
            com.github.tjake.jlama.safetensors.SafeTensorSupport.readTensorInfoMap(safeBuf, Optional.of(metadata))

        return Weights(metadata, tensorInfoMap, safeBuf.slice(), Optional.empty())
    }

    @Throws(IOException::class)
    fun detectModel(configFile: File?): ModelType {
        val rootNode: JsonNode = om.readTree(configFile)
        require(rootNode.has("model_type")) { "Config missing model_type field." }

        return ModelType.valueOf(rootNode.get("model_type").textValue().toUpperCase())
    }

    fun loadWeights(baseDir: File): WeightLoader {
        if (Files.exists(
                Paths.get(
                    baseDir.getAbsolutePath(),
                    SafeTensorIndex.MODEL_INDEX_JSON
                )
            )
        ) return SafeTensorIndex.loadWithWeights(
            baseDir.toPath()
        )

        if (Files.exists(
                Paths.get(
                    baseDir.getAbsolutePath(),
                    SafeTensorIndex.SINGLE_MODEL_NAME
                )
            )
        ) return SafeTensorIndex.loadSingleFile(
            baseDir.toPath(),
            SafeTensorIndex.SINGLE_MODEL_NAME
        )

        throw IllegalArgumentException("No safetensor model found in: " + baseDir)
    }

    fun isModelLocal(modelRoot: Path): Boolean {
        if (Files.exists(modelRoot.resolve(SafeTensorIndex.SINGLE_MODEL_NAME))) return true
        try {
            if (Files.exists(modelRoot.resolve(SafeTensorIndex.MODEL_INDEX_JSON))) {
                val index: SafeTensorIndex = om.readValue(
                    modelRoot.resolve(SafeTensorIndex.MODEL_INDEX_JSON).toFile(),
                    SafeTensorIndex::class.java
                )

                for (file in index.weightFileMap.values()) {
                    if (!Files.exists(modelRoot.resolve(file))) {
                        return false
                    }
                }

                return true
            }
        } catch (e: IOException) {
            com.github.tjake.jlama.safetensors.SafeTensorSupport.logger.error("Error reading model index", e)
            return false
        }

        return false
    }

    @Throws(IOException::class)
    fun loadTokenizer(modelRoot: Path): TokenizerModel {
        val tokenizerJson: File = modelRoot.resolve("tokenizer.json").toFile()
        Preconditions.checkArgument(tokenizerJson.exists(), "No tokenizer.json found in " + modelRoot)

        val rootNode: JsonNode = om.readTree(tokenizerJson)
        require(rootNode.has("model")) { "Json missing 'model' key" }

        val model: TokenizerModel = om.treeToValue(rootNode.get("model"), TokenizerModel::class.java)

        if (rootNode.has("added_tokens") && rootNode.get("added_tokens") != null) {
            val addedTokens: List<Map<String?, Object?>?>? =
                om.convertValue(rootNode.get("added_tokens"), List::class.java)
            model.setAddedTokens(addedTokens)
        }

        if (rootNode.has("pre_tokenizer") && rootNode.get("pre_tokenizer") != null) model.setPreTokenizer(
            om.treeToValue(rootNode.get("pre_tokenizer"), TokenizerModel.PreTokenizer::class.java)
        )

        if (rootNode.has("normalizer") && rootNode.get("normalizer") != null) model.setNormalizer(
            om.treeToValue(rootNode.get("normalizer"), TokenizerModel.Normalizer::class.java)
        )

        val tokenizerConfigJson: File = modelRoot.resolve("tokenizer_config.json").toFile()
        if (tokenizerConfigJson.exists()) {
            val configNode: JsonNode = om.readTree(tokenizerConfigJson)
            if (configNode.has("legacy")) model.setLegacy(configNode.get("legacy").asBoolean())

            if (configNode.has("chat_template")) {
                val chatTemplateNode: JsonNode = configNode.get("chat_template")
                val promptTemplates: Map<String?, String?> = HashMap()
                if (chatTemplateNode.isTextual()) {
                    promptTemplates.put("default", chatTemplateNode.asText())
                } else if (chatTemplateNode.isArray()) {
                    val chatTemplates: List<Map<String?, String?>?> =
                        om.convertValue(chatTemplateNode, List::class.java)
                    for (chatTemplate in chatTemplates) {
                        if (chatTemplate!!.containsKey("name") && chatTemplate.containsKey("template")) {
                            promptTemplates.put(chatTemplate.get("name"), chatTemplate.get("template"))
                        } else {
                            throw IllegalArgumentException("Invalid chat_template format")
                        }
                    }
                } else {
                    throw IllegalArgumentException("Invalid chat_template format")
                }

                model.setPromptTemplates(promptTemplates)
            }

            if (configNode.has("eos_token")) {
                model.setEosToken(configNode.get("eos_token").asText())
            }

            if (configNode.has("bos_token")) {
                model.setBosToken(configNode.get("bos_token").asText())
            }
        }

        return model
    }

    @Throws(IOException::class)
    fun quantizeModel(
        modelRoot: Path,
        modelQuantization: DType,
        skipLayerPrefixes: Array<String?>?,
        dropLayerPrefixes: Array<String?>?,
        outputRoot: Optional<Path?>
    ): Path {
        val tmp: File = File.createTempFile("safe", "tensor")
        tmp.deleteOnExit()
        val wl: WeightLoader = com.github.tjake.jlama.safetensors.SafeTensorSupport.loadWeights(modelRoot.toFile())
        val writtenInfo: Map<String?, Object?> = HashMap()

        RandomAccessFile(tmp, "rw").use { raf ->
            val tensors: Map<String?, TensorInfo?> = wl.tensorInfoMap()
            for (e in tensors.entrySet()) {
                var drop = false
                if (dropLayerPrefixes != null) {
                    for (dropLayerPrefix in dropLayerPrefixes) {
                        if (e.getKey().startsWith(dropLayerPrefix)) {
                            com.github.tjake.jlama.safetensors.SafeTensorSupport.logger.info("Dropping layer: " + e.getKey())
                            drop = true
                        }
                    }
                }

                if (drop) continue

                wl.load(e.getKey()).use { tr ->
                    var skipQ = false
                    if (skipLayerPrefixes != null) {
                        for (skipLayerPrefix in skipLayerPrefixes) {
                            if (e.getKey().contains(skipLayerPrefix)) {
                                com.github.tjake.jlama.safetensors.SafeTensorSupport.logger.info("Skipping quantization of layer: " + e.getKey())
                                skipQ = true
                                break
                            }
                        }
                    }

                    val t: AbstractTensor = if (skipQ) tr else tr.quantize(modelQuantization)
                    when (t.dType()) {
                        F32, BF16, F16 -> writtenInfo.put(e.getKey(), t.save(raf.getChannel()))
                        Q4 -> {
                            writtenInfo.put(e.getKey(), t.save(raf.getChannel()))
                            writtenInfo.put(
                                e.getKey() + ".qb",
                                (t as Q4ByteBufferTensor).getBlockF().save(raf.getChannel())
                            )
                        }

                        Q5 -> {
                            writtenInfo.put(e.getKey(), t.save(raf.getChannel()))
                            writtenInfo.put(
                                e.getKey() + ".qb",
                                (t as Q5ByteBufferTensor).getBlockF().save(raf.getChannel())
                            )
                            // FIXME: Need to add b5 bits
                            throw UnsupportedOperationException("TODO")
                        }

                        I8 -> {
                            writtenInfo.put(e.getKey(), t.save(raf.getChannel()))
                            writtenInfo.put(
                                e.getKey() + ".qb",
                                (t as Q8ByteBufferTensor).getBlockF().save(raf.getChannel())
                            )
                        }

                        else -> throw UnsupportedOperationException("" + t.dType() + " not implemented")
                    }
                }
            }
        }
        // Now create the output file
        val baseDirName = modelRoot.getName(modelRoot.getNameCount() - 1).toString()
        val parentPath: Path = modelRoot.getParent()

        val qPath: Path = outputRoot.orElseGet({
            Paths.get(
                parentPath.toString(),
                baseDirName.toString() + "-J" + modelQuantization.name()
            )
        })
        val qDir: File = qPath.toFile()
        qDir.mkdirs()

        // Copy config.json and tokenizer.json
        Files.copy(modelRoot.resolve("config.json"), qPath.resolve("config.json"))
        Files.copy(modelRoot.resolve("tokenizer.json"), qPath.resolve("tokenizer.json"))
        Files.copy(modelRoot.resolve("README.md"), qPath.resolve("README.md"))

        // Copy README.md and add jlama header
        com.github.tjake.jlama.safetensors.SafeTensorSupport.addJlamaHeader(baseDirName, qPath.resolve("README.md"))

        if (Files.exists(modelRoot.resolve("tokenizer_config.json"))) Files.copy(
            modelRoot.resolve("tokenizer_config.json"),
            qPath.resolve("tokenizer_config.json")
        )

        RandomAccessFile(qPath.resolve("model.safetensors").toFile(), "rw").use { raf ->
            val header: ByteArray = om.writeValueAsBytes(writtenInfo)
            val hsize = ByteArray(Long.BYTES)
            ByteBuffer.wrap(hsize).order(ByteOrder.LITTLE_ENDIAN).putLong(header.size)
            raf.write(hsize)
            raf.write(header)
            Files.copy(tmp.toPath(), object : OutputStream() {
                @Override
                @Throws(IOException::class)
                fun write(b: Int) {
                    raf.write(b)
                }

                @Override
                @Throws(IOException::class)
                fun write(b: ByteArray?) {
                    raf.write(b)
                }

                @Override
                @Throws(IOException::class)
                fun write(b: ByteArray?, off: Int, len: Int) {
                    raf.write(b, off, len)
                }
            })
        }
        return qPath
    }

    @Throws(IOException::class)
    private fun addJlamaHeader(modelName: String, readmePath: Path?) {
        val cleanName: String? = modelName.replaceAll("_", "/")
        val header: String? = String.format(
            ("# Quantized Version of %s \n\n"
                    + "This model is a quantized variant of the %s model, optimized for use with Jlama, a Java-based inference engine. "
                    + "The quantization process reduces the model's size and improves inference speed, while maintaining high accuracy "
                    + "for efficient deployment in production environments.\n\n"
                    + "For more information on Jlama, visit the [Jlama GitHub repository](https://github.com/tjake/jlama).\n\n"
                    + "---\n\n"),
            cleanName,
            cleanName
        )
        val readme = String(Files.readAllBytes(readmePath))
        var startMeta = false
        var finishedMeta = false
        var linenum = 0
        val finalReadme = StringBuilder()
        for (line in readme.split("\n")) {
            var line = line
            if (linenum++ == 0) {
                if (line.startsWith("---")) {
                    startMeta = true
                } else {
                    finalReadme.append(header)
                }
            } else if (startMeta && !finishedMeta && line.startsWith("---")) {
                finishedMeta = true
                line += "\n\n" + header
            }
            finalReadme.append(line).append("\n")
        }

        Files.write(readmePath, finalReadme.toString().getBytes())
    }

    @Throws(IOException::class)
    fun maybeDownloadModel(modelDir: String?, fullModelName: String, progressReporter: ProgressReporter?): File {
        val parts: Array<String?> = fullModelName.split("/")
        require(!(parts.size == 0 || parts.size > 2)) { "Model must be in the form owner/name" }

        val owner: String?
        val name: String?

        if (parts.size == 1) {
            owner = null
            name = fullModelName
        } else {
            owner = parts[0]
            name = parts[1]
        }

        return com.github.tjake.jlama.safetensors.SafeTensorSupport.maybeDownloadModel(
            modelDir,
            Optional.ofNullable(owner),
            name,
            true,
            Optional.empty(),
            Optional.empty(),
            Optional.ofNullable(progressReporter)
        )
    }

    @Throws(IOException::class)
    fun maybeDownloadModel(modelDir: String?, fullModelName: String): File {
        return com.github.tjake.jlama.safetensors.SafeTensorSupport.maybeDownloadModel(modelDir, fullModelName, null)
    }

    fun constructLocalModelPath(modelDir: String?, owner: String?, modelName: String?): Path {
        return Paths.get(modelDir, owner.toString() + "_" + modelName)
    }

    var FINISHED_MARKER: String = ".finished"

    /**
     * Download a model from HuggingFace and return the path to the model directory
     *
     * @param modelDir The directory to save the model to
     * @param modelOwner The owner of the HF model (if any)
     * @param modelName The name of the HF model
     * @param downloadWeights Include the weights or leave them out
     * @param optionalBranch The branch of the model to download
     * @param optionalAuthHeader The authorization header to use for the request
     * @param optionalProgressReporter A consumer to report download progress
     * @return The path to the downloaded model directory
     * @throws IOException
     */
    @Throws(IOException::class)
    fun maybeDownloadModel(
        modelDir: String?,
        modelOwner: Optional<String?>,
        modelName: String?,
        downloadWeights: Boolean,
        optionalBranch: Optional<String?>,
        optionalAuthHeader: Optional<String?>?,
        optionalProgressReporter: Optional<ProgressReporter?>?
    ): File {
        val localModelDir: Path = com.github.tjake.jlama.safetensors.SafeTensorSupport.constructLocalModelPath(
            modelDir,
            modelOwner.orElse("na"),
            modelName
        )
        // Check if the model is already downloaded
        if (Files.exists(localModelDir.resolve(com.github.tjake.jlama.safetensors.SafeTensorSupport.FINISHED_MARKER))) {
            return localModelDir.toFile()
        }

        val hfModel: String? = modelOwner.map({ mo -> mo.toString() + "/" + modelName }).orElse(modelName)
        val modelInfoStream: InputStream? = HttpSupport.getResponse(
            "https://huggingface.co/api/models/" + hfModel + "/tree/" + optionalBranch.orElse("main"),
            optionalAuthHeader,
            Optional.empty()
        ).left
        val modelInfo: String = HttpSupport.readInputStream(modelInfoStream)

        if (modelInfo == null) {
            throw IOException("No valid model found or trying to access a restricted model (please include correct access token)")
        }

        val allFiles: List<String?> = com.github.tjake.jlama.safetensors.SafeTensorSupport.parseFileList(modelInfo)
        if (allFiles.isEmpty()) {
            throw IOException("No valid model found")
        }

        val tensorFiles: List<String?> = ArrayList()
        var hasSafetensor = false
        for (currFile in allFiles) {
            val f = currFile!!.toLowerCase()
            if ((f.contains("safetensor") && !f.contains("consolidated"))
                || f.contains("readme")
                || f.equals("config.json")
                || f.contains("tokenizer")
            ) {
                if (f.contains("safetensor")) {
                    hasSafetensor = true
                }

                if (!downloadWeights && f.contains("safetensor")) {
                    continue
                }

                tensorFiles.add(currFile)
            }
        }

        if (!hasSafetensor) {
            throw IOException("Model is not available in safetensor format")
        }

        Files.createDirectories(localModelDir)

        for (currFile in tensorFiles) {
            HttpSupport.downloadFile(
                hfModel,
                currFile,
                optionalBranch,
                optionalAuthHeader,
                Optional.empty(),
                localModelDir.resolve(currFile),
                optionalProgressReporter
            )
        }

        // When fully downloaded, create a .finished file
        Files.createFile(localModelDir.resolve(com.github.tjake.jlama.safetensors.SafeTensorSupport.FINISHED_MARKER))

        return localModelDir.toFile()
    }

    @Throws(IOException::class)
    private fun parseFileList(modelInfo: String?): List<String?> {
        val fileList: List<String?> = ArrayList()

        val objectMapper: ObjectMapper = ObjectMapper()
        val siblingsNode: JsonNode = objectMapper.readTree(modelInfo)
        if (siblingsNode.isArray()) {
            for (siblingNode in siblingsNode) {
                val rFilename: String? = siblingNode.path("path").asText()
                fileList.add(rFilename)
            }
        }

        return fileList
    }
}
