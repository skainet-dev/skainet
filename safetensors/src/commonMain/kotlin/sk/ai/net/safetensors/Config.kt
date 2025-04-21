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
package sk.ai.net.safetensors

import com.github.tjake.jlama.math.ActivationFunction
import kotlin.concurrent.Volatile

class Config(
    val contextLength: Int,
    val embeddingLength: Int,
    val hiddenLength: Int,
    val numberOfHeads: Int,
    val numberOfKeyValueHeads: Int,
    val numberOfLayers: Int,
    val layerNormEps: Float,
    val vocabularySize: Int,
    val bosToken: Int,
    eosTokens: List<Int>,
    //activationFunction: ActivationFunction.Type?,
    ropeFreqsTheta: Double?,
    ropeScalingFactor: Double?,
    classifcationLabels: Map<String, Int>,
    headSize: Int,
    finalLogitSoftCapping: Float?,
    attnLogitSoftCapping: Float?,
    residualMultiplier: Float?,
    attentionMultiplier: Float?,
    embeddingMultiplier: Float?,
    logitMultiplier: Float?
) {
    val attentionLength: Int
    val headSize: Int
    val activationFunction: ActivationFunction.Type?
    val headGroupSize: Int
    val kvLength: Int
    val isGQA: Boolean
    val finalLogitSoftCapping: Float?
    val attnLogitSoftCapping: Float?
    val residualMultiplier: Float?
    val attentionMultiplier: Float?
    val embeddingMultiplier: Float?
    val logitMultiplier: Float?
    val eosTokens: List<Int?>?
    val classifcationLabels: BiMap<String, Int>



    constructor(
        contextLength: Int,
        embeddingLength: Int,
        hiddenLength: Int,
        numberOfHeads: Int,
        numberOfKeyValueHeads: Int,
        numberOfLayers: Int,
        layerNormEps: Float,
        vocabularySize: Int,
        bosToken: Int,
        eosToken: List<Integer?>?,
        activationFunction: ActivationFunction.Type?,
        ropeFreqsTheta: Double?,
        ropeScalingFactor: Double?,
        headSize: Integer?,
        attnLogitSoftCapping: Float?,
        finalLogitSoftCapping: Float?
    ) : this(
        contextLength,
        embeddingLength,
        hiddenLength,
        numberOfHeads,
        numberOfKeyValueHeads,
        numberOfLayers,
        layerNormEps,
        vocabularySize,
        bosToken,
        eosToken,
        activationFunction,
        ropeFreqsTheta,
        ropeScalingFactor,
        null,
        if (headSize == null) embeddingLength / numberOfHeads else headSize,
        attnLogitSoftCapping,
        finalLogitSoftCapping,
        null,
        null,
        null,
        null
    )

    constructor(
        contextLength: Int,
        embeddingLength: Int,
        hiddenLength: Int,
        numberOfHeads: Int,
        numberOfKeyValueHeads: Int,
        numberOfLayers: Int,
        layerNormEps: Float,
        vocabularySize: Int,
        bosToken: Int,
        eosToken: List<Int>,
        //activationFunction: ActivationFunction.Type?,
        ropeFreqsTheta: Double?,
        ropeScalingFactor: Double?
    ) : this(
        contextLength,
        embeddingLength,
        hiddenLength,
        numberOfHeads,
        numberOfKeyValueHeads,
        numberOfLayers,
        layerNormEps,
        vocabularySize,
        bosToken,
        eosToken,
        //activationFunction,
        ropeFreqsTheta,
        ropeScalingFactor,
        null,
        embeddingLength / numberOfHeads,
        null,
        null,
        null,
        null,
        null,
        null
    )

    constructor(
        contextLength: Int,
        embeddingLength: Int,
        hiddenLength: Int,
        numberOfHeads: Int,
        numberOfKeyValueHeads: Int,
        numberOfLayers: Int,
        layerNormEps: Float,
        vocabularySize: Int,
        bosToken: Int,
        eosToken: List<Int>,
        //activationFunction: ActivationFunction.Type?,
        ropeFreqsTheta: Double?,
        ropeScalingFactor: Double?,
        residualMultiplier: Float?,
        attentionMultiplier: Float?,
        embeddingMultiplier: Float?,
        logitMultiplier: Float?
    ) : this(
        contextLength,
        embeddingLength,
        hiddenLength,
        numberOfHeads,
        numberOfKeyValueHeads,
        numberOfLayers,
        layerNormEps,
        vocabularySize,
        bosToken,
        eosToken,
        activationFunction,
        ropeFreqsTheta,
        ropeScalingFactor,
        null,
        embeddingLength / numberOfHeads,
        null,
        null,
        residualMultiplier,
        attentionMultiplier,
        embeddingMultiplier,
        logitMultiplier
    )

    constructor(
        contextLength: Int,
        embeddingLength: Int,
        hiddenLength: Int,
        numberOfHeads: Int,
        numberOfKeyValueHeads: Int,
        numberOfLayers: Int,
        layerNormEps: Float,
        vocabularySize: Int,
        bosToken: Int,
        eosToken: List<Int>,
        //activationFunction: ActivationFunction.Type?,
        ropeFreqsTheta: Double?,
        ropeScalingFactor: Double?,
        classifcationLabels: Map<String, Int>
    ) : this(
        contextLength,
        embeddingLength,
        hiddenLength,
        numberOfHeads,
        numberOfKeyValueHeads,
        numberOfLayers,
        layerNormEps,
        vocabularySize,
        bosToken,
        eosToken,
        activationFunction,
        ropeFreqsTheta,
        ropeScalingFactor,
        classifcationLabels,
        embeddingLength / numberOfHeads,
        null,
        null,
        null,
        null,
        null,
        null
    )

    init {
        this.attentionLength = numberOfHeads * headSize
        this.eosTokens = eosTokens
        this.headSize = headSize
        this.headGroupSize = numberOfHeads / numberOfKeyValueHeads
        this.kvLength = numberOfKeyValueHeads * headSize
        this.isGQA = numberOfKeyValueHeads < numberOfHeads
        //this.activationFunction = activationFunction

        this.classifcationLabels = classifcationLabels
//            if (classifcationLabels == null) Optional.empty() else Optional.of(ImmutableBiMap.copyOf(classifcationLabels))

        this.finalLogitSoftCapping = finalLogitSoftCapping
        this.attnLogitSoftCapping = attnLogitSoftCapping
        this.residualMultiplier = residualMultiplier
        this.attentionMultiplier = attentionMultiplier
        this.embeddingMultiplier = embeddingMultiplier
        this.logitMultiplier = logitMultiplier

        // Set default values
        this.dctx = DistributedContext.builder(this).build()
    }

    fun setDistributedContext(dctx: DistributedContext?) {
        this.dctx = dctx
    }

    fun setWorkingDirectory(workingDirectory: File?) {
        if (workingDirectory == null) {
            this.workingDirectory = Files.createTempDir()
            this.workingDirectory.deleteOnExit()
        } else {
            Preconditions.checkArgument(workingDirectory.isDirectory())
            this.workingDirectory = workingDirectory
        }
    }

    fun workingDirectory(): Optional<File?> {
        return Optional.ofNullable(this.workingDirectory)
    }

    fun dctx(): DistributedContext? {
        return dctx
    }

    fun maybeMapToGroupHead(head: Int): Int {
        if (!isGQA) return head
        return Math.floorDiv(head, headGroupSize)
    }

    val isClassifier: Boolean
        get() = classifcationLabels.isPresent()
}
