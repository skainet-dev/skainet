package sk.ainet.core.tensor.factory

import sk.ainet.core.tensor.*
import sk.ainet.core.tensor.backend.CpuTensorFP32
import kotlin.test.*

/**
 * Comprehensive tests for TensorFactoryRegistry functionality.
 * Tests factory registration, lookup, thread safety, and error handling.
 */
class TensorFactoryRegistryTest {

    @BeforeTest
    fun setup() {
        // Reset registry to clean state before each test
        TensorFactoryInitializer.reset()
    }

    @AfterTest
    fun cleanup() {
        // Clean up after each test
        TensorFactoryInitializer.reset()
    }

    @Test
    fun testFactoryRegistration() {
        // Initially no factories should be registered
        assertEquals(0, TensorFactoryRegistry.getRegisteredDTypes().size, "Registry should start empty")
        assertFalse(TensorFactoryRegistry.hasFactory(FP32), "FP32 factory should not be registered initially")

        // Register FP32 factory
        TensorFactoryRegistry.registerFactory<FP32>(FP32TensorFactory)

        // Verify registration
        assertTrue(TensorFactoryRegistry.hasFactory(FP32), "FP32 factory should be registered")
        assertEquals(1, TensorFactoryRegistry.getRegisteredDTypes().size, "Should have 1 registered factory")

        val registeredTypes = TensorFactoryRegistry.getRegisteredDTypes()
        assertTrue(registeredTypes.any { it.simpleName == "FP32" }, "FP32 should be in registered types")
    }

    @Test
    fun testMultipleFactoryRegistration() {
        // Register multiple factories
        TensorFactoryRegistry.registerFactory<FP32>(FP32TensorFactory)
        TensorFactoryRegistry.registerFactory<Int8>(Int8TensorFactory)
        TensorFactoryRegistry.registerFactory<Int32>(Int32TensorFactory)

        // Verify all are registered
        assertTrue(TensorFactoryRegistry.hasFactory(FP32), "FP32 factory should be registered")
        assertTrue(TensorFactoryRegistry.hasFactory(Int8), "Int8 factory should be registered")
        assertTrue(TensorFactoryRegistry.hasFactory(Int32), "Int32 factory should be registered")
        assertEquals(3, TensorFactoryRegistry.getRegisteredDTypes().size, "Should have 3 registered factories")
    }

    @Test
    fun testFactoryLookup() {
        // Register factories
        TensorFactoryRegistry.registerFactory<FP32>(FP32TensorFactory)
        TensorFactoryRegistry.registerFactory<Int8>(Int8TensorFactory)

        // Test successful lookup
        assertTrue(TensorFactoryRegistry.hasFactory(FP32), "Should find FP32 factory")
        assertTrue(TensorFactoryRegistry.hasFactory(Int8), "Should find Int8 factory")

        // Test unsuccessful lookup for unregistered type
        assertFalse(TensorFactoryRegistry.hasFactory(Int32), "Should not find unregistered Int32 factory")
    }

    @Test
    fun testCreateTensorWithRegisteredFactory() {
        // Register factory
        TensorFactoryRegistry.registerFactory<FP32>(FP32TensorFactory)

        val shape = Shape(2, 2) // 4 floats = 16 bytes
        val floats = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
        val bytes = ByteArray(16)
        for (i in floats.indices) {
            val bits = floats[i].toBits()
            bytes[i * 4 + 0] = (bits and 0xFF).toByte()
            bytes[i * 4 + 1] = ((bits shr 8) and 0xFF).toByte()
            bytes[i * 4 + 2] = ((bits shr 16) and 0xFF).toByte()
            bytes[i * 4 + 3] = ((bits shr 24) and 0xFF).toByte()
        }

        val tensor = TensorFactoryRegistry.createTensor(FP32, shape, bytes)
        assertNotNull(tensor, "Tensor should be created successfully")
        assertEquals(shape, tensor.shape, "Tensor should have correct shape")
    }

    @Test
    fun testCreateTensorWithUnregisteredFactory() {
        // Don't register any factories
        val shape = Shape(2, 2)
        val bytes = ByteArray(16)

        // Should throw exception for unregistered DType
        assertFailsWith<IllegalArgumentException> {
            TensorFactoryRegistry.createTensor(FP32, shape, bytes)
        }
    }

    @Test
    fun testFactoryOverride() {
        // Create a custom factory for testing override behavior
        val customFactory = object : TensorFactoryRegistry.TensorFromBytesFactory<FP32, Float> {
            override fun fromByteArray(shape: Shape, data: ByteArray, littleEndian: Boolean): Tensor<FP32, Float> {
                // Simple implementation for testing
                val floatArray = ByteArrayConverter.convertBytesToFloatArray(data)
                return CpuTensorFP32(shape, floatArray)
            }
        }

        // Register original factory
        TensorFactoryRegistry.registerFactory<FP32>(FP32TensorFactory)
        assertTrue(TensorFactoryRegistry.hasFactory(FP32), "FP32 factory should be registered")

        // Override with custom factory
        TensorFactoryRegistry.registerFactory<FP32>(customFactory)
        assertTrue(TensorFactoryRegistry.hasFactory(FP32), "FP32 factory should still be registered")
        assertEquals(1, TensorFactoryRegistry.getRegisteredDTypes().size, "Should still have 1 factory (overridden)")
    }

    @Test
    fun testGetRegisteredDTypes() {
        // Start with empty registry
        var registeredTypes = TensorFactoryRegistry.getRegisteredDTypes()
        assertEquals(0, registeredTypes.size, "Should start with no registered types")

        // Register factories one by one
        TensorFactoryRegistry.registerFactory<FP32>(FP32TensorFactory)
        registeredTypes = TensorFactoryRegistry.getRegisteredDTypes()
        assertEquals(1, registeredTypes.size, "Should have 1 registered type")
        assertTrue(registeredTypes.any { it.simpleName == "FP32" }, "Should contain FP32")

        TensorFactoryRegistry.registerFactory<Int8>(Int8TensorFactory)
        registeredTypes = TensorFactoryRegistry.getRegisteredDTypes()
        assertEquals(2, registeredTypes.size, "Should have 2 registered types")
        assertTrue(registeredTypes.any { it.simpleName == "FP32" }, "Should contain FP32")
        assertTrue(registeredTypes.any { it.simpleName == "Int8" }, "Should contain Int8")

        TensorFactoryRegistry.registerFactory<Int32>(Int32TensorFactory)
        registeredTypes = TensorFactoryRegistry.getRegisteredDTypes()
        assertEquals(3, registeredTypes.size, "Should have 3 registered types")
        assertTrue(registeredTypes.any { it.simpleName == "FP32" }, "Should contain FP32")
        assertTrue(registeredTypes.any { it.simpleName == "Int8" }, "Should contain Int8")
        assertTrue(registeredTypes.any { it.simpleName == "Int32" }, "Should contain Int32")
    }

    @Test
    fun testRegistryReset() {
        // Register some factories
        TensorFactoryRegistry.registerFactory<FP32>(FP32TensorFactory)
        TensorFactoryRegistry.registerFactory<Int8>(Int8TensorFactory)
        assertEquals(2, TensorFactoryRegistry.getRegisteredDTypes().size, "Should have 2 registered factories")

        // Reset registry
        TensorFactoryInitializer.reset()

        // Verify registry is empty
        assertEquals(0, TensorFactoryRegistry.getRegisteredDTypes().size, "Registry should be empty after reset")
        assertFalse(TensorFactoryRegistry.hasFactory(FP32), "FP32 factory should not be registered after reset")
        assertFalse(TensorFactoryRegistry.hasFactory(Int8), "Int8 factory should not be registered after reset")
    }

    @Test
    fun testErrorHandlingForNullData() {
        TensorFactoryRegistry.registerFactory<FP32>(FP32TensorFactory)
        val shape = Shape(2, 2)

        // Test with empty byte array
        val emptyBytes = ByteArray(0)
        assertFailsWith<IllegalArgumentException> {
            TensorFactoryRegistry.createTensor(FP32, shape, emptyBytes)
        }
    }

    @Test
    fun testErrorHandlingForInvalidShape() {
        TensorFactoryRegistry.registerFactory<FP32>(FP32TensorFactory)
        val bytes = ByteArray(16) // 4 floats worth of data

        // Test with zero dimensions
        assertFailsWith<IllegalArgumentException> {
            val invalidShape = Shape(0, 4)
            TensorFactoryRegistry.createTensor(FP32, invalidShape, bytes)
        }

        // Test with negative dimensions
        assertFailsWith<IllegalArgumentException> {
            val invalidShape = Shape(-1, 4)
            TensorFactoryRegistry.createTensor(FP32, invalidShape, bytes)
        }
    }

    @Test
    fun testEdgeCaseMinimalTensor() {
        TensorFactoryRegistry.registerFactory<FP32>(FP32TensorFactory)

        // Create minimal valid tensor (1x1 with single float)
        val shape = Shape(1, 1)
        val floats = floatArrayOf(42.0f)
        val bytes = ByteArray(4)
        val bits = floats[0].toBits()
        bytes[0] = (bits and 0xFF).toByte()
        bytes[1] = ((bits shr 8) and 0xFF).toByte()
        bytes[2] = ((bits shr 16) and 0xFF).toByte()
        bytes[3] = ((bits shr 24) and 0xFF).toByte()

        // This should work for minimal tensors
        val tensor = TensorFactoryRegistry.createTensor(FP32, shape, bytes)
        assertNotNull(tensor, "Minimal tensor should be created successfully")
        assertEquals(shape, tensor.shape, "Minimal tensor should have correct shape")
        assertEquals(1, tensor.shape.volume, "Minimal tensor should have volume of 1")
    }

    @Test
    fun testRegistryStateConsistency() {
        // Test that registry maintains consistent state
        val initialCount = TensorFactoryRegistry.getRegisteredDTypes().size

        // Register factory
        TensorFactoryRegistry.registerFactory<FP32>(FP32TensorFactory)
        assertEquals(initialCount + 1, TensorFactoryRegistry.getRegisteredDTypes().size, "Count should increase by 1")
        assertTrue(TensorFactoryRegistry.hasFactory(FP32), "hasFactory should return true")

        // Register same factory again (override)
        TensorFactoryRegistry.registerFactory<FP32>(FP32TensorFactory)
        assertEquals(
            initialCount + 1,
            TensorFactoryRegistry.getRegisteredDTypes().size,
            "Count should remain same after override"
        )
        assertTrue(TensorFactoryRegistry.hasFactory(FP32), "hasFactory should still return true")
    }
}