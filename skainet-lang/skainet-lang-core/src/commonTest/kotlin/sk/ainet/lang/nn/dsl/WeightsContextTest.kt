package sk.ainet.lang.nn.dsl

import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertNotNull

class WeightsContextTest {

    @Test
    fun weightsContextDsl() {
        println("[DEBUG_LOG] Starting WeightsContext DSL test")
        

        // Simple test to check if from and ones work in weights context
        val testNetwork = context<FP32, Float> {
            sequential {
                input(1)
                dense(2) {
                    weights { shape ->
                        println("[DEBUG_LOG] Creating weights with shape: $shape")
                        from(0.5f, 0.3f)
                    }
                    bias { shape ->
                        println("[DEBUG_LOG] Creating bias with shape: $shape")
                        ones()
                    }
                }
            }
        }
        
        println("[DEBUG_LOG] Network created successfully: ${testNetwork.name}")
        
        // Verify that the network was created
        assertNotNull(testNetwork, "Network should not be null")
        println("[DEBUG_LOG] WeightsContext DSL test completed successfully")
    }
}