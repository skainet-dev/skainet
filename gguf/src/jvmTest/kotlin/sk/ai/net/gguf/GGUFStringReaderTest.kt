package sk.ai.net.gguf

import junit.framework.Assert.assertEquals
import kotlinx.io.asSource
import kotlinx.io.buffered
import org.junit.Test


class GGUFReaderTest {

    @Test
    fun testReadMetadataFields() {
        javaClass.getResourceAsStream("/skainet-small.gguf").use { inputStream ->

            val reader = GGUFReader(inputStream.asSource().buffered())

            // Verify the 'model_name' metadata is correct
            val modelName = reader.getString("model_name")
            assertEquals("model_name should match", "skainet-small", modelName)

            // Verify the 'authors' metadata list is correct
            val authorsList = reader.getStringList("authors")
            assertEquals("authors list should match", 2, authorsList.size)
            //assertEquals (listOf("Alice", "Bob"), authorsList, "authors list should match")
        }
    }
}
