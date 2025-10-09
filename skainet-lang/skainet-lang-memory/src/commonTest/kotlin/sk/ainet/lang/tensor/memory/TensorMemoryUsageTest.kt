package sk.ainet.lang.tensor.memory

import sk.ainet.lang.tensor.Shape
import kotlin.test.*

/**
 * Focused memory usage tests for the refactored tensor data classes.
 * This test directly uses the tensor classes without depending on DenseTensorDataFactory.
 */
class TensorMemoryUsageTest {
    
    @Test
    fun testFloatTensorMemoryUsage() {
        println("[DEBUG_LOG] === Float Tensor Memory Analysis ===")
        
        val shapes = listOf(
            Shape(intArrayOf(100)),           // 1D: 100 elements = 400 bytes
            Shape(intArrayOf(50, 50)),        // 2D: 2500 elements = 10,000 bytes
            Shape(intArrayOf(10, 10, 10))     // 3D: 1000 elements = 4,000 bytes
        )
        
        shapes.forEach { shape ->
            val expectedBytes = shape.volume * 4L // 4 bytes per Float
            val data = FloatArray(shape.volume) { 0.0f }
            val tensor = DenseFloatTensorArray(shape, data)
            
            println("[DEBUG_LOG] Float tensor ${shape.dimensions.contentToString()}: ${shape.volume} elements, $expectedBytes bytes")
            
            // Test functionality
            val zeroIndices = IntArray(shape.dimensions.size) { 0 }
            tensor.set(*zeroIndices, value = 3.14f)
            assertEquals(3.14f, tensor.get(*zeroIndices), 0.001f)
            
            // Test edge case - last element
            val lastIndices = IntArray(shape.dimensions.size) { i -> shape.dimensions[i] - 1 }
            tensor.set(*lastIndices, value = 2.71f)
            assertEquals(2.71f, tensor.get(*lastIndices), 0.001f)
        }
    }
    
    @Test
    fun testByteTensorMemoryUsage() {
        println("[DEBUG_LOG] === Byte Tensor Memory Analysis ===")
        
        val shapes = listOf(
            Shape(intArrayOf(1000)),          // 1D: 1000 elements = 1,000 bytes
            Shape(intArrayOf(100, 100)),      // 2D: 10,000 elements = 10,000 bytes
            Shape(intArrayOf(20, 20, 25))     // 3D: 10,000 elements = 10,000 bytes
        )
        
        shapes.forEach { shape ->
            val expectedBytes = shape.volume * 1L // 1 byte per Byte
            val data = ByteArray(shape.volume) { 0 }
            val tensor = DenseByteTensorArray(shape, data)
            
            println("[DEBUG_LOG] Byte tensor ${shape.dimensions.contentToString()}: ${shape.volume} elements, $expectedBytes bytes")
            
            // Test functionality
            val zeroIndices = IntArray(shape.dimensions.size) { 0 }
            tensor.set(*zeroIndices, value = 127)
            assertEquals(127.toByte(), tensor.get(*zeroIndices))
            
            tensor.set(*zeroIndices, value = -128)
            assertEquals((-128).toByte(), tensor.get(*zeroIndices))
        }
    }
    
    @Test
    fun testInt4TensorMemoryEfficiency() {
        println("[DEBUG_LOG] === Int4 Tensor Memory Analysis ===")
        
        val elementCounts = listOf(100, 1000, 10000)
        
        elementCounts.forEach { count ->
            val shape = Shape(intArrayOf(count))
            val bytesNeeded = (count + 1) / 2 // 2 Int4 values per byte
            val data = ByteArray(bytesNeeded) { 0 }
            val tensor = DenseInt4TensorArray(shape, data)
            
            val regularIntBytes = count * 4L
            val compressionRatio = (regularIntBytes * 10 / bytesNeeded) / 10.0
            
            println("[DEBUG_LOG] Int4 tensor: $count elements")
            println("[DEBUG_LOG]   Int4 storage: $bytesNeeded bytes")
            println("[DEBUG_LOG]   Int32 would use: $regularIntBytes bytes")
            println("[DEBUG_LOG]   Compression ratio: ${compressionRatio}:1")
            println("[DEBUG_LOG]   Memory savings: ${((regularIntBytes - bytesNeeded) * 1000 / regularIntBytes).toInt() / 10.0}%")
            
            // Test functionality across the valid range
            tensor.set(0, value = 7)   // Max positive
            assertEquals(7, tensor.get(0))
            
            tensor.set(0, value = -8)  // Max negative
            assertEquals(-8, tensor.get(0))
            
            tensor.set(0, value = 0)   // Zero
            assertEquals(0, tensor.get(0))
            
            // Test multiple values in same byte
            if (count >= 2) {
                tensor.set(0, value = 3)
                tensor.set(1, value = -5)
                assertEquals(3, tensor.get(0))
                assertEquals(-5, tensor.get(1))
            }
        }
    }
    
    @Test
    fun testTernaryTensorMemoryEfficiency() {
        println("[DEBUG_LOG] === Ternary Tensor Memory Analysis ===")
        
        val elementCounts = listOf(100, 1000, 10000)
        
        elementCounts.forEach { count ->
            val shape = Shape(intArrayOf(count))
            val bytesNeeded = DenseTernaryTensorArray.requiredBytesForElements(count)
            val data = ByteArray(bytesNeeded) { 0 }
            val tensor = DenseTernaryTensorArray(shape, data)
            
            val regularByteBytes = count * 1L
            val regularIntBytes = count * 4L
            val compressionVsByte = (regularByteBytes * 10 / bytesNeeded) / 10.0
            val compressionVsInt = (regularIntBytes * 10 / bytesNeeded) / 10.0
            
            println("[DEBUG_LOG] Ternary tensor: $count elements")
            println("[DEBUG_LOG]   Ternary storage: $bytesNeeded bytes (${(count * 2.0 / 8).let { if (it > it.toInt()) it.toInt() + 1 else it.toInt() }} bytes for ${count * 2} bits)")
            println("[DEBUG_LOG]   Byte would use: $regularByteBytes bytes")
            println("[DEBUG_LOG]   Int32 would use: $regularIntBytes bytes")
            println("[DEBUG_LOG]   Compression vs Byte: ${compressionVsByte}:1")
            println("[DEBUG_LOG]   Compression vs Int32: ${compressionVsInt}:1")
            println("[DEBUG_LOG]   Memory savings vs Byte: ${((regularByteBytes - bytesNeeded) * 1000 / regularByteBytes).toInt() / 10.0}%")
            
            // Test functionality across the ternary range
            tensor.set(0, value = 1)   // Positive
            assertEquals(1.toByte(), tensor.get(0))
            
            tensor.set(0, value = 0)   // Zero
            assertEquals(0.toByte(), tensor.get(0))
            
            tensor.set(0, value = -1)  // Negative
            assertEquals((-1).toByte(), tensor.get(0))
            
            // Test multiple values in same byte (4 ternary values per byte)
            if (count >= 4) {
                tensor.set(0, value = 1)
                tensor.set(1, value = 0)
                tensor.set(2, value = -1)
                tensor.set(3, value = 1)
                assertEquals(1.toByte(), tensor.get(0))
                assertEquals(0.toByte(), tensor.get(1))
                assertEquals((-1).toByte(), tensor.get(2))
                assertEquals(1.toByte(), tensor.get(3))
            }
        }
    }
    
    @Test
    fun testMemoryComparisonSummary() {
        println("[DEBUG_LOG] === Memory Comparison Summary ===")
        
        val elementCount = 10000
        val shape = Shape(intArrayOf(elementCount))
        
        // Float tensor
        val floatData = FloatArray(elementCount) { 0.0f }
        val floatTensor = DenseFloatTensorArray(shape, floatData)
        val floatBytes = elementCount * 4L
        
        // Byte tensor  
        val byteData = ByteArray(elementCount) { 0 }
        val byteTensor = DenseByteTensorArray(shape, byteData)
        val byteBytes = elementCount * 1L
        
        // Int4 tensor
        val int4BytesNeeded = (elementCount + 1) / 2
        val int4Data = ByteArray(int4BytesNeeded) { 0 }
        val int4Tensor = DenseInt4TensorArray(shape, int4Data)
        
        // Ternary tensor
        val ternaryBytesNeeded = DenseTernaryTensorArray.requiredBytesForElements(elementCount)
        val ternaryData = ByteArray(ternaryBytesNeeded) { 0 }
        val ternaryTensor = DenseTernaryTensorArray(shape, ternaryData)
        
        println("[DEBUG_LOG] For $elementCount elements:")
        println("[DEBUG_LOG] Float (FP32):     $floatBytes bytes    (baseline)")
        println("[DEBUG_LOG] Byte (Int8):      $byteBytes bytes     (${((floatBytes - byteBytes) * 1000 / floatBytes).toInt() / 10.0}% savings)")
        println("[DEBUG_LOG] Int4:             $int4BytesNeeded bytes     (${((floatBytes - int4BytesNeeded) * 1000 / floatBytes).toInt() / 10.0}% savings)")
        println("[DEBUG_LOG] Ternary (2-bit):  $ternaryBytesNeeded bytes    (${((floatBytes - ternaryBytesNeeded) * 1000 / floatBytes).toInt() / 10.0}% savings)")
        
        println("[DEBUG_LOG] Bits per element:")
        println("[DEBUG_LOG] Float: 32 bits")
        println("[DEBUG_LOG] Byte:  8 bits") 
        println("[DEBUG_LOG] Int4:  4 bits")
        println("[DEBUG_LOG] Ternary: 2 bits")
        
        // Test that all tensors work correctly
        floatTensor.set(0, value = 123.45f)
        assertEquals(123.45f, floatTensor.get(0), 0.001f)
        
        byteTensor.set(0, value = 123)
        assertEquals(123.toByte(), byteTensor.get(0))
        
        int4Tensor.set(0, value = 7)
        assertEquals(7, int4Tensor.get(0))
        
        ternaryTensor.set(0, value = 1)
        assertEquals(1.toByte(), ternaryTensor.get(0))
    }
    
    @Test
    fun testErrorHandlingAndBoundaries() {
        println("[DEBUG_LOG] === Error Handling and Boundary Tests ===")
        
        // Test Int4 boundaries
        val int4Shape = Shape(intArrayOf(10))
        val int4Data = ByteArray(5)
        val int4Tensor = DenseInt4TensorArray(int4Shape, int4Data)
        
        // Valid range: -8 to 7
        int4Tensor.set(0, value = -8)
        assertEquals(-8, int4Tensor.get(0))
        
        int4Tensor.set(0, value = 7)
        assertEquals(7, int4Tensor.get(0))
        
        // Invalid values should throw
        assertFailsWith<IllegalArgumentException> {
            int4Tensor.set(0, value = 8)
        }
        assertFailsWith<IllegalArgumentException> {
            int4Tensor.set(0, value = -9)
        }
        
        // Test Ternary boundaries
        val ternaryShape = Shape(intArrayOf(8))
        val ternaryBytes = DenseTernaryTensorArray.requiredBytesForElements(8)
        val ternaryData = ByteArray(ternaryBytes)
        val ternaryTensor = DenseTernaryTensorArray(ternaryShape, ternaryData)
        
        // Valid range: -1, 0, 1
        ternaryTensor.set(0, value = -1)
        assertEquals((-1).toByte(), ternaryTensor.get(0))
        
        ternaryTensor.set(0, value = 0)
        assertEquals(0.toByte(), ternaryTensor.get(0))
        
        ternaryTensor.set(0, value = 1)
        assertEquals(1.toByte(), ternaryTensor.get(0))
        
        // Invalid values should throw
        assertFailsWith<IllegalArgumentException> {
            ternaryTensor.set(0, value = 2)
        }
        assertFailsWith<IllegalArgumentException> {
            ternaryTensor.set(0, value = -2)
        }
        
        println("[DEBUG_LOG] All boundary tests passed!")
    }
}