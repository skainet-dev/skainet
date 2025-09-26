package sk.ainet.core.tensor.factory

import sk.ainet.core.tensor.*
import kotlin.test.*

/**
 * Tests for the tensor factory system including registry, byte conversion, and factory implementations.
 */
class TensorFactoryTest {
    
    @BeforeTest
    fun setup() {
        // Reset and initialize factories before each test
        TensorFactoryInitializer.reset()
        TensorFactoryInitializer.initializeFactories()
    }
    
    @AfterTest
    fun cleanup() {
        // Clean up after each test
        TensorFactoryInitializer.reset()
    }
    
    @Test
    fun testFactoryInitialization() {
        assertTrue(TensorFactoryInitializer.isInitialized(), "Factory should be initialized")
        assertEquals(3, TensorFactoryInitializer.getRegisteredFactoryCount(), "Should have 3 registered factories")
        
        val registeredTypes = TensorFactoryRegistry.getRegisteredDTypes()
        assertTrue(registeredTypes.any { it.simpleName == "FP32" }, "FP32 factory should be registered")
        assertTrue(registeredTypes.any { it.simpleName == "Int8" }, "Int8 factory should be registered")
        assertTrue(registeredTypes.any { it.simpleName == "Int32" }, "Int32 factory should be registered")
    }
    
    @Test
    fun testFP32TensorCreation() {
        val shape = Shape(2, 2) // 2x2 tensor = 4 floats = 16 bytes
        
        // Create test data: [1.0f, 2.0f, 3.0f, 4.0f] in little-endian bytes
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
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Tensor should have correct shape")
    }
    
    @Test
    fun testInt8TensorCreation() {
        val shape = Shape(3, 2) // 3x2 tensor = 6 bytes
        val bytes = byteArrayOf(1, 2, 3, 4, 5, 6)
        
        val tensor = TensorFactoryRegistry.createTensor(Int8, shape, bytes)
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Tensor should have correct shape")
    }
    
    @Test
    fun testInt32TensorCreation() {
        val shape = Shape(2, 1) // 2x1 tensor = 2 ints = 8 bytes
        
        // Create test data: [100, 200] in little-endian bytes
        val ints = intArrayOf(100, 200)
        val bytes = ByteArray(8)
        for (i in ints.indices) {
            bytes[i * 4 + 0] = (ints[i] and 0xFF).toByte()
            bytes[i * 4 + 1] = ((ints[i] shr 8) and 0xFF).toByte()
            bytes[i * 4 + 2] = ((ints[i] shr 16) and 0xFF).toByte()
            bytes[i * 4 + 3] = ((ints[i] shr 24) and 0xFF).toByte()
        }
        
        val tensor = TensorFactoryRegistry.createTensor(Int32, shape, bytes)
        assertNotNull(tensor, "Tensor should be created")
        assertEquals(shape, tensor.shape, "Tensor should have correct shape")
    }
    
    @Test
    fun testByteArrayConverter() {
        // Test float conversion
        val floatBytes = byteArrayOf(0x00, 0x00, 0x80.toByte(), 0x3F) // 1.0f in little-endian
        val floats = ByteArrayConverter.convertBytesToFloatArray(floatBytes)
        assertEquals(1, floats.size, "Should have 1 float")
        assertEquals(1.0f, floats[0], 0.001f, "Should convert to 1.0f")
        
        // Test int conversion
        val intBytes = byteArrayOf(0x64, 0x00, 0x00, 0x00) // 100 in little-endian
        val ints = ByteArrayConverter.convertBytesToIntArray(intBytes)
        assertEquals(1, ints.size, "Should have 1 int")
        assertEquals(100, ints[0], "Should convert to 100")
        
        // Test byte conversion (identity)
        val originalBytes = byteArrayOf(1, 2, 3, 4)
        val copiedBytes = ByteArrayConverter.convertBytesToByteArray(originalBytes)
        assertContentEquals(originalBytes, copiedBytes, "Byte arrays should be identical")
    }
    
    @Test
    fun testFactoryRegistryFunctionality() {
        // Test hasFactory method
        assertTrue(TensorFactoryRegistry.hasFactory(FP32), "Should have FP32 factory")
        assertTrue(TensorFactoryRegistry.hasFactory(Int8), "Should have Int8 factory")
        assertTrue(TensorFactoryRegistry.hasFactory(Int32), "Should have Int32 factory")
        
        // Test getRegisteredDTypes
        val registeredTypes = TensorFactoryRegistry.getRegisteredDTypes()
        assertEquals(3, registeredTypes.size, "Should have 3 registered types")
    }
    
    @Test
    fun testInputValidation() {
        val shape = Shape(2, 2) // Needs 16 bytes for FP32
        val wrongSizeBytes = byteArrayOf(1, 2, 3, 4) // Only 4 bytes
        
        assertFailsWith<IllegalArgumentException> {
            FP32TensorFactory.fromGGUFData(shape, wrongSizeBytes)
        }
        
        assertFailsWith<IllegalArgumentException> {
            Int32TensorFactory.fromGGUFData(shape, wrongSizeBytes)
        }
    }
}