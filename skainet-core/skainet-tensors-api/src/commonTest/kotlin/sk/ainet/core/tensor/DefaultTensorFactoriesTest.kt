package sk.ainet.core.tensor

import sk.ainet.core.tensor.backend.CpuBackend
import sk.ainet.core.tensor.backend.CpuBackendInt8
import sk.ainet.core.tensor.backend.CpuBackendInt32
import kotlin.test.*

/**
 * Tests for DefaultTensorFactories to verify factory initialization
 * and identify any IllegalStateException issues
 */
class DefaultTensorFactoriesTest {

    @Test
    fun testFP32FactoryInitialization() {
        try {
            DefaultTensorFactories.setFP32Factory(CpuBackend())
            val factory = DefaultTensorFactories.getFP32Factory()
            val tensor = factory.zeros(Shape(2, 3))
            assertNotNull(tensor)
            assertEquals(Shape(2, 3), tensor.shape)
            println("[DEBUG_LOG] FP32 factory initialized successfully")
        } catch (e: IllegalStateException) {
            println("[DEBUG_LOG] FP32 factory not initialized: ${e.message}")
            fail("FP32 factory not initialized: ${e.message}")
        }
    }

    @Test
    fun testInt8FactoryInitialization() {
        try {
            val factory = DefaultTensorFactories.getInt8Factory()
            val tensor = factory.zeros(Shape(2, 3))
            assertNotNull(tensor)
            assertEquals(Shape(2, 3), tensor.shape)
            println("[DEBUG_LOG] Int8 factory initialized successfully")
        } catch (e: IllegalStateException) {
            println("[DEBUG_LOG] Int8 factory not initialized: ${e.message}")
            fail("Int8 factory not initialized: ${e.message}")
        }
    }

    @Test
    fun testInt32FactoryInitialization() {
        try {
            val factory = DefaultTensorFactories.getInt32Factory()
            val tensor = factory.zeros(Shape(2, 3))
            assertNotNull(tensor)
            assertEquals(Shape(2, 3), tensor.shape)
            println("[DEBUG_LOG] Int32 factory initialized successfully")
        } catch (e: IllegalStateException) {
            println("[DEBUG_LOG] Int32 factory not initialized: ${e.message}")
            fail("Int32 factory not initialized: ${e.message}")
        }
    }

    @Test
    fun testAllFactoryTypes() {
        println("[DEBUG_LOG] Testing all factory types...")
        
        // Force initialization by creating backend instances
        try {
            CpuBackend()
            println("[DEBUG_LOG] FP32 backend instantiated")
        } catch (e: Exception) {
            println("[DEBUG_LOG] Failed to instantiate FP32 backend: ${e.message}")
        }
        
        try {
            CpuBackendInt8()
            println("[DEBUG_LOG] Int8 backend instantiated")
        } catch (e: Exception) {
            println("[DEBUG_LOG] Failed to instantiate Int8 backend: ${e.message}")
        }
        
        try {
            CpuBackendInt32()
            println("[DEBUG_LOG] Int32 backend instantiated")
        } catch (e: Exception) {
            println("[DEBUG_LOG] Failed to instantiate Int32 backend: ${e.message}")
        }
        
        // This test will show us which factories are missing
        val results = mutableListOf<String>()
        
        try {
            DefaultTensorFactories.getFP32Factory()
            results.add("FP32: OK")
        } catch (e: IllegalStateException) {
            results.add("FP32: MISSING - ${e.message}")
        }
        
        try {
            DefaultTensorFactories.getInt8Factory()
            results.add("Int8: OK")
        } catch (e: IllegalStateException) {
            results.add("Int8: MISSING - ${e.message}")
        }
        
        try {
            DefaultTensorFactories.getInt32Factory()
            results.add("Int32: OK")
        } catch (e: IllegalStateException) {
            results.add("Int32: MISSING - ${e.message}")
        }
        
        results.forEach { println("[DEBUG_LOG] $it") }
        
        // We'll fail only if all factories are missing
        val missingCount = results.count { it.contains("MISSING") }
        if (missingCount == results.size) {
            fail("All factories are missing!")
        }
    }
}