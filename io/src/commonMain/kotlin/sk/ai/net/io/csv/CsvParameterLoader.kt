package sk.ai.net.io.csv

import kotlinx.serialization.json.Json
import okio.BufferedSource
import okio.use
import sk.ai.net.Shape

import sk.ai.net.Tensor
import sk.ai.net.impl.DoublesTensor
import sk.ai.net.io.ParametersLoader

class CsvParametersLoader(private val handleSource: () -> BufferedSource) :
    ParametersLoader {
    override suspend fun load(onTensorLoaded: (String, Tensor) -> Unit) {
        handleSource().use { source ->
            // Initialize Json object
            val json = Json { ignoreUnknownKeys = true }
            // Deserialize JSON to Kotlin objects
            json.decodeFromString<List<Parameter>>(source.readUtf8()).also { values ->
                values.forEach { (name, array) ->
                    val tensor = DoublesTensor(Shape(*array.shape.toIntArray()), array.values.toDoubleArray())
                    onTensorLoaded(name, tensor)
                }
            }
        }
    }
}