package sk.ai.net.io.csv

import kotlinx.serialization.json.Json
import okio.Path
import sk.ai.net.Shape

import sk.ai.net.Tensor
import sk.ai.net.impl.DoublesTensor
import sk.ai.net.io.ParametersLoader

class CsvParametersLoader(private val fileSystem: okio.FileSystem, private val modelWeightPath: Path) :
    ParametersLoader {
    override suspend fun load(onTensorLoaded: (String, Tensor) -> Unit) {
        fileSystem.read(modelWeightPath) {
            // Initialize Json object
            val json = Json { ignoreUnknownKeys = true }

            // Deserialize JSON to Kotlin objects
            json.decodeFromString<List<ArrayValues>>(this.readUtf8()).also { values ->
                values.forEach { (name, array) ->
                    val tensor = DoublesTensor(Shape(*array.shape.toIntArray()), array.values.toDoubleArray())
                    onTensorLoaded(name, tensor)
                }
            }
        }
    }
}