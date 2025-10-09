package sk.ainet.lang.tensor.memory

import sk.ainet.lang.tensor.Shape
import kotlin.test.*

/**
 * Simple memory manager to track memory allocations for testing purposes.
 * This is a basic implementation to measure theoretical memory usage.
 */
class SimpleMemoryManager {
    private var totalAllocatedBytes = 0L
    private val allocations = mutableListOf<AllocationRecord>()
    
    data class AllocationRecord(
        val name: String,
        val sizeBytes: Long,
        val timestamp: Long = 0L // Simplified for multiplatform compatibility
    )
    
    fun allocateFloatArray(size: Int, name: String = "FloatArray"): FloatArray {
        val sizeInBytes = size * 4L // 4 bytes per Float
        totalAllocatedBytes += sizeInBytes
        allocations.add(AllocationRecord(name, sizeInBytes))
        return FloatArray(size)
    }
    
    fun allocateByteArray(size: Int, name: String = "ByteArray"): ByteArray {
        val sizeInBytes = size * 1L // 1 byte per Byte
        totalAllocatedBytes += sizeInBytes
        allocations.add(AllocationRecord(name, sizeInBytes))
        return ByteArray(size)
    }
    
    fun getTotalAllocatedBytes(): Long = totalAllocatedBytes
    
    fun getAllocations(): List<AllocationRecord> = allocations.toList()
    
    fun reset() {
        totalAllocatedBytes = 0L
        allocations.clear()
    }
    
    fun printMemoryReport() {
        println("[DEBUG_LOG] === Memory Allocation Report ===")
        println("[DEBUG_LOG] Total allocated: $totalAllocatedBytes bytes")
        allocations.forEach { record ->
            println("[DEBUG_LOG] ${record.name}: ${record.sizeBytes} bytes")
        }
        println("[DEBUG_LOG] ================================")
    }
}

class MemoryUsageTest {
    
    private val memoryManager = SimpleMemoryManager()
    
    @BeforeTest
    fun setUp() {
        memoryManager.reset()
    }
    
    @Test
    fun testDenseMemoryFloatTensorDataMemoryUsage() {
        // Test different tensor shapes and their memory consumption
        val shapes = listOf(
            Shape(intArrayOf(10)),           // 1D: 10 elements
            Shape(intArrayOf(10, 10)),       // 2D: 100 elements  
            Shape(intArrayOf(10, 10, 10)),   // 3D: 1000 elements
            Shape(intArrayOf(5, 20, 20, 3))  // 4D: 6000 elements
        )
        
        shapes.forEach { shape ->
            val expectedBytes = shape.volume * 4L // 4 bytes per Float
            val data = memoryManager.allocateFloatArray(shape.volume, "Float tensor ${shape.dimensions.contentToString()}")
            val tensor = DenseFloatTensorArray(shape, data)
            
            println("[DEBUG_LOG] Float tensor ${shape.dimensions.contentToString()}: expected $expectedBytes bytes")
            
            // Test basic functionality
            val indices = IntArray(shape.dimensions.size) { 0 }
            tensor.set(*indices, value = 3.14f)
            assertEquals(3.14f, tensor.get(*indices), 0.001f)
        }
        
        memoryManager.printMemoryReport()
    }
    
    @Test
    fun testDenseMemoryByteTensorDataMemoryUsage() {
        val shapes = listOf(
            Shape(intArrayOf(100)),          // 1D: 100 elements
            Shape(intArrayOf(50, 50)),       // 2D: 2500 elements
            Shape(intArrayOf(20, 20, 20))    // 3D: 8000 elements
        )
        
        shapes.forEach { shape ->
            val expectedBytes = shape.volume * 1L // 1 byte per Byte
            val data = memoryManager.allocateByteArray(shape.volume, "Byte tensor ${shape.dimensions.contentToString()}")
            val tensor = DenseByteTensorArray(shape, data)
            
            println("[DEBUG_LOG] Byte tensor ${shape.dimensions.contentToString()}: expected $expectedBytes bytes")
            
            // Test basic functionality
            val indices = IntArray(shape.dimensions.size) { 0 }
            tensor.set(*indices, value = 42.toByte())
            assertEquals(42.toByte(), tensor.get(*indices))
        }
        
        memoryManager.printMemoryReport()
    }
    
    @Test
    fun testDenseMemoryInt4TensorDataMemoryUsage() {
        val shapes = listOf(
            Shape(intArrayOf(16)),           // 1D: 16 elements = 8 bytes
            Shape(intArrayOf(10, 10)),       // 2D: 100 elements = 50 bytes
            Shape(intArrayOf(8, 8, 8))       // 3D: 512 elements = 256 bytes
        )
        
        shapes.forEach { shape ->
            val expectedBytes = (shape.volume + 1) / 2 // 2 Int4 values per byte
            val actualBytesNeeded = (shape.volume + 1) / 2
            val data = memoryManager.allocateByteArray(actualBytesNeeded, "Int4 tensor ${shape.dimensions.contentToString()}")
            val tensor = DenseInt4TensorArray(shape, data)
            
            println("[DEBUG_LOG] Int4 tensor ${shape.dimensions.contentToString()}: ${shape.volume} elements, expected $expectedBytes bytes")
            
            // Test basic functionality with Int4 range (-8 to 7)
            val indices = IntArray(shape.dimensions.size) { 0 }
            tensor.set(*indices, value = 7)
            assertEquals(7, tensor.get(*indices))
            
            tensor.set(*indices, value = -8)
            assertEquals(-8, tensor.get(*indices))
            
            // Test memory efficiency compared to regular integers
            val regularIntBytes = shape.volume * 4L
            val compressionRatio = regularIntBytes.toDouble() / expectedBytes
            println("[DEBUG_LOG] Int4 compression ratio vs Int32: ${(compressionRatio * 10).toInt() / 10.0}:1")
        }
        
        memoryManager.printMemoryReport()
    }
    
    @Test
    fun testDenseTernaryTensorDataMemoryUsage() {
        val shapes = listOf(
            Shape(intArrayOf(16)),           // 1D: 16 elements = 4 bytes (2 bits per element)
            Shape(intArrayOf(12, 12)),       // 2D: 144 elements = 36 bytes
            Shape(intArrayOf(8, 8, 8))       // 3D: 512 elements = 128 bytes
        )
        
        shapes.forEach { shape ->
            val expectedBytes = DenseTernaryTensorArray.requiredBytesForElements(shape.volume)
            val data = memoryManager.allocateByteArray(expectedBytes, "Ternary tensor ${shape.dimensions.contentToString()}")
            val tensor = DenseTernaryTensorArray(shape, data)
            
            println("[DEBUG_LOG] Ternary tensor ${shape.dimensions.contentToString()}: ${shape.volume} elements, expected $expectedBytes bytes")
            
            // Test basic functionality with ternary values (-1, 0, 1)
            val indices = IntArray(shape.dimensions.size) { 0 }
            
            tensor.set(*indices, value = 1)
            assertEquals(1.toByte(), tensor.get(*indices))
            
            tensor.set(*indices, value = 0)
            assertEquals(0.toByte(), tensor.get(*indices))
            
            tensor.set(*indices, value = -1)
            assertEquals((-1).toByte(), tensor.get(*indices))
            
            // Test memory efficiency compared to regular bytes
            val regularByteBytes = shape.volume * 1L
            val compressionRatio = regularByteBytes.toDouble() / expectedBytes
            println("[DEBUG_LOG] Ternary compression ratio vs Byte: ${(compressionRatio * 10).toInt() / 10.0}:1")
        }
        
        memoryManager.printMemoryReport()
    }
    
    @Test
    fun testMemoryComparisonAcrossTypes() {
        val elementCount = 1000
        val shape = Shape(intArrayOf(elementCount))
        
        println("[DEBUG_LOG] === Memory Comparison for $elementCount elements ===")
        
        // Float tensor
        val floatData = memoryManager.allocateFloatArray(elementCount, "Float comparison")
        val floatTensor = DenseFloatTensorArray(shape, floatData)
        val floatBytes = elementCount * 4L
        println("[DEBUG_LOG] Float tensor: $floatBytes bytes")
        
        // Byte tensor
        val byteData = memoryManager.allocateByteArray(elementCount, "Byte comparison")
        val byteTensor = DenseByteTensorArray(shape, byteData)
        val byteBytes = elementCount * 1L
        println("[DEBUG_LOG] Byte tensor: $byteBytes bytes")
        
        // Int4 tensor
        val int4BytesNeeded = (elementCount + 1) / 2
        val int4Data = memoryManager.allocateByteArray(int4BytesNeeded, "Int4 comparison")
        val int4Tensor = DenseInt4TensorArray(shape, int4Data)
        println("[DEBUG_LOG] Int4 tensor: $int4BytesNeeded bytes")
        
        // Ternary tensor
        val ternaryBytesNeeded = DenseTernaryTensorArray.requiredBytesForElements(elementCount)
        val ternaryData = memoryManager.allocateByteArray(ternaryBytesNeeded, "Ternary comparison")
        val ternaryTensor = DenseTernaryTensorArray(shape, ternaryData)
        println("[DEBUG_LOG] Ternary tensor: $ternaryBytesNeeded bytes")
        
        // Calculate compression ratios
        val int4Savings = ((floatBytes - int4BytesNeeded) * 1000.0 / floatBytes).toInt() / 10.0
        val ternarySavingsVsFloat = ((floatBytes - ternaryBytesNeeded) * 1000.0 / floatBytes).toInt() / 10.0
        val ternarySavingsVsByte = ((byteBytes - ternaryBytesNeeded) * 1000.0 / byteBytes).toInt() / 10.0
        println("[DEBUG_LOG] Int4 saves $int4Savings% vs Float")
        println("[DEBUG_LOG] Ternary saves $ternarySavingsVsFloat% vs Float")
        println("[DEBUG_LOG] Ternary saves $ternarySavingsVsByte% vs Byte")
        
        memoryManager.printMemoryReport()
    }
    
    @Test
    fun testErrorHandling() {
        val shape = Shape(intArrayOf(5, 5))
        val data = ByteArray(25)
        val tensor = DenseByteTensorArray(shape, data)
        
        // Test index out of bounds
        assertFailsWith<IllegalArgumentException> {
            tensor[10, 0]
        }
        
        assertFailsWith<IllegalArgumentException> {
            tensor[0, 10]
        }
        
        // Test wrong number of indices
        assertFailsWith<IllegalArgumentException> {
            tensor[0]
        }
        
        assertFailsWith<IllegalArgumentException> {
            tensor[0, 0, 0]
        }
    }
    
    @Test
    fun testInt4ValueConstraints() {
        val shape = Shape(intArrayOf(10))
        val data = ByteArray(5) // 10 Int4 values need 5 bytes
        val tensor = DenseInt4TensorArray(shape, data)
        
        // Test valid range
        tensor.set(0, value = 7)
        assertEquals(7, tensor[0])
        
        tensor.set(0, value = -8)
        assertEquals(-8, tensor[0])
        
        // Test invalid values
        assertFailsWith<IllegalArgumentException> {
            tensor.set(0, value = 8)
        }
        
        assertFailsWith<IllegalArgumentException> {
            tensor.set(0, value = -9)
        }
    }
    
    @Test
    fun testTernaryValueConstraints() {
        val shape = Shape(intArrayOf(8))
        val requiredBytes = DenseTernaryTensorArray.requiredBytesForElements(8)
        val data = ByteArray(requiredBytes)
        val tensor = DenseTernaryTensorArray(shape, data)
        
        // Test valid values
        tensor.set(0, value = 1)
        assertEquals(1.toByte(), tensor[0])
        
        tensor.set(0, value = 0)
        assertEquals(0.toByte(), tensor[0])
        
        tensor.set(0, value = -1)
        assertEquals((-1).toByte(), tensor[0])
        
        // Test invalid values
        assertFailsWith<IllegalArgumentException> {
            tensor.set(0, value = 2)
        }
        
        assertFailsWith<IllegalArgumentException> {
            tensor.set(0, value = -2)
        }
    }
}