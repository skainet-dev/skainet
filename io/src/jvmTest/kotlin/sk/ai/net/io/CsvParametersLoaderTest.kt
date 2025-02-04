package sk.ai.net.io

import kotlinx.coroutines.runBlocking
import okio.buffer
import okio.source
import org.junit.Test
import sk.ai.net.Shape
import sk.ai.net.dsl.network
import sk.ai.net.io.csv.CsvParametersLoader
import sk.ai.net.io.mapper.NamesBasedValuesModelMapper
import sk.ai.net.nn.reflection.flattenParams
import sk.ai.net.nn.reflection.summary
import sk.ai.net.nn.reflection.toVisualString
import java.io.InputStream
import kotlin.test.assertEquals

class CsvParametersLoaderTest {


    @Test
    fun `test csv load with mapper by names`() {

        val sineModule = network {
            input(1)
            dense(16) {
                activation = { it }
            }
            dense(16) {
                activation = { it }
            }
            dense(1)
        }

        print(sineModule.summary(Shape(1)))


        print(sineModule.toVisualString())

        javaClass.getResourceAsStream("/sinus-approximator.json")?.use { inputStream: InputStream ->
            //assertNotNull(inputStream, "Test resource file not found!")


            val parametersLoader = CsvParametersLoader { inputStream.source().buffer() }

            val mapper = NamesBasedValuesModelMapper()

            runBlocking {
                parametersLoader.load { name, tensor ->
                    mapper.mapToModel(sineModule, mapOf(name to tensor))
                }
                val params = flattenParams(sineModule)
                assertEquals(params.size, 6)
            }
            assertEquals(1, 1)
        }


    }
}