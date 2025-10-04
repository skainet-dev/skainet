import sk.ainet.core.tensor.FP32
import sk.ainet.core.tensor.Shape
import sk.ainet.nn.dsl.*
import kotlin.test.Test


class SinNetworkShapeTest {

    @Test
    fun weigthsContextDsl() {
        // Simple test to check if fromArray works in weights context
        val testNetwork = network<FP32, Float> {
            input(1)
            dense(2) {
                weights { shape ->
                    //fromList(listOf(0.5f, 0.3f))
                    from(0.5f, 0.3f)
                }
                bias {
                    ones()
                    //fromArray(0.1f, 0.2f)
                }
            }
        }
        println("Network created: ${testNetwork.name}")
    }
}