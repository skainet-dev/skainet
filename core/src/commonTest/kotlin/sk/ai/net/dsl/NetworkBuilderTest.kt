package sk.ai.net.dsl

import sk.ai.net.nn.activations.ReLU
import sk.ai.net.nn.activations.Softmax
import kotlin.test.Test
import kotlin.test.assertNotNull

class NetworkBuilderTest {
    @Test
    fun testNetworkWithStages() {
        val mnistNetwork = network {
            sequential {
                stage("conv1") {
                    conv2d {
                        outChannels = 16
                        kernelSize = 5
                        stride = 1
                        padding = 2
                    }
                    activation("relu", ReLU()::forward)
                    maxPool2d {
                        kernelSize = 2
                        stride = 2
                    }
                }
                stage("conv2") {
                    conv2d {
                        outChannels = 32
                        kernelSize = 5
                        stride = 1
                        padding = 2
                    }
                    activation("relu", ReLU()::forward)
                    maxPool2d {
                        kernelSize = 2
                        stride = 2
                    }
                }
                stage("flatten") {
                    flatten()
                }
                stage("dense") {
                    dense {
                        units = 128
                    }
                    activation("relu", ReLU()::forward)
                }
                stage("output") {
                    dense {
                        units = 10
                    }
                    activation("softmax", Softmax(1)::forward)
                }
            }
        }

        assertNotNull(mnistNetwork)
    }
}
