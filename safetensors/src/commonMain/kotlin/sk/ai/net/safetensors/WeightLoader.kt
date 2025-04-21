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

//import com.github.tjake.jlama.model.DistributedContext
import com.github.tjake.jlama.safetensors.DType
import com.github.tjake.jlama.safetensors.TensorInfo

interface WeightLoader : AutoCloseable {
    fun metadata(): Map<String?, String?>?

    fun tensorInfoMap(): Map<String?, TensorInfo?>?

    fun isWeightPresent(name: String?): Boolean {
        return tensorInfoMap()!!.containsKey(name)
    }

    fun load(name: String?): AbstractTensor? {
        return load(name, false, false)
    }

    fun load(name: String?,sparseRows: Boolean, sparseColumns: Boolean): Tensor

    //fun load(name: String?, dctx: DistributedContext?, sparseRows: Boolean, sparseColumns: Boolean): AbstractTensor?

    val modelDType: DType?
}
