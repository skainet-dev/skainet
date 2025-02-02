package sk.ai.net.gguf

import org.junit.Assert
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotSame
import org.junit.Test
import sk.ai.net.gguf.utils.reshape
import java.io.File

class GGUFReaderTest {

    @Test
    fun test_reshape() {
        assertEquals(
            listOf(listOf(1, 2), listOf(3, 4)),
            listOf(1, 2, 3, 4).reshape(2, 2)
        )

        assertEquals(
            listOf(listOf(1, 2, 3, 4)),
            listOf(1, 2, 3, 4).reshape(1, 4)
        )

        assertEquals(
            listOf(listOf(1),listOf(2),listOf(3),listOf(4)),
            listOf(1, 2, 3, 4).reshape(4, 1)
        )

        assertNotSame(
            listOf(listOf(1),listOf(2),listOf(3),listOf(4)),
            List(384 * 1536) { it + 0.02387242 }.reshape(384, 1536)
        )
    }


    /**
     * You may run this test to read from a gguf model and print model metadata in console output.
     * Change the file path to read from your own model.
     *
     * To find a gguf model, you can look from here:
     * https://huggingface.co/models?library=gguf
     */
    @Test
    fun `test GGUFReader with valid model file`() {
        val resourcesDirectory = File("src/test/resources")
        val modelFile = File("$resourcesDirectory/test_experiment.gguf")
        val modelFilePath = modelFile.path

        Assert.assertTrue("Model file should exist: $modelFilePath", modelFile.exists())

        val reader = GGUFReader(modelFilePath)

        // List all key-value pairs in a columned format
        println("Key-Value Pairs:")
        val maxKeyLength = reader.fields.keys.maxOf { it.length }
        for ((key, field) in reader.fields) {
            val value = if (field.types[0] == GGUFValueType.STRING && field.types.size == 1) {
                String(
                    (field.parts[field.data[0]] as List<UByte>).toUByteArray().toByteArray(),
                    Charsets.UTF_8
                )
            } else {
                field.parts[field.data[0]]
            }

            println("${key.padEnd(maxKeyLength)} : $value")
        }
        println("----")

        // List all tensors
        println("Tensors:")
        val tensorInfoFormat = "%-30s | Shape: %-15s | Size: %-12s | Quantization: %s"
        println(tensorInfoFormat.format("Tensor Name", "Shape", "Size", "Quantization"))
        println("-".repeat(80))
        for (tensor in reader.tensors) {
            val shapeStr = tensor.shape.joinToString("x")
            val sizeStr = tensor.nElements.toString()
            val quantizationStr = tensor.tensorType.name
            println(tensorInfoFormat.format(tensor.name, shapeStr, sizeStr, quantizationStr))
        }
    }
}
