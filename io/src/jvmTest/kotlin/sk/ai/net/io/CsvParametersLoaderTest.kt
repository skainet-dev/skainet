package sk.ai.net.io

import junit.framework.TestCase.assertTrue
import kotlinx.coroutines.runBlocking
import okio.buffer
import okio.source
import org.junit.Test
import sk.ai.net.Shape
import sk.ai.net.dsl.network
import sk.ai.net.impl.DoublesTensor
import sk.ai.net.io.csv.CsvParametersLoader
import sk.ai.net.io.mapper.NamesBasedValuesModelMapper
import sk.ai.net.nn.Module
import sk.ai.net.nn.activations.ReLU
import sk.ai.net.nn.reflection.flattenParams
import sk.ai.net.nn.reflection.summary
import sk.ai.net.nn.reflection.toVisualString
import java.io.InputStream
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.test.assertEquals

class CsvParametersLoaderTest {

    fun Module.of(angle: Double): Double =
        (this.forward(DoublesTensor(Shape(1), listOf(angle.toDouble()).toDoubleArray())) as DoublesTensor)[0]


    @Test
    fun `test csv load with mapper by names`() {

        val sineModule: Module = network {
            input(1)
            dense(16) {
                activation = ReLU()::forward
            }
            dense(16) {
                activation = ReLU()::forward
            }
            dense(1)
        }

        print(sineModule.summary(Shape(1)))


        print(sineModule.toVisualString())

        javaClass.getResourceAsStream("/sinus-approximator.json")?.use { inputStream: InputStream ->
            val parametersLoader = CsvParametersLoader { inputStream.source().buffer() }

            val mapper = NamesBasedValuesModelMapper()

            runBlocking {
                parametersLoader.load { name, tensor ->
                    mapper.mapToModel(sineModule, mapOf(name to tensor))
                }
                val params = flattenParams(sineModule)
                assertEquals(params.size, 6)

                print(sineModule.summary(Shape(1)))
                print(sineModule.toVisualString())
                assertTrue(abs(sineModule.of(PI / 2.0) - sin(PI / 2.0)) < 0.01)
            }
        }
    }
}