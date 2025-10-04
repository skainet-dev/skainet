package sk.ainet.core.tensor

import sk.ainet.core.tensor.backend.CpuTensorFP32
import sk.ainet.core.tensor.backend.CpuTensorInt8
import sk.ainet.core.tensor.backend.CpuTensorInt32
import kotlin.test.Test
import kotlin.test.assertTrue

class ToStringTest {

    @Test
    fun testCpuTensorFP32ToString() {
        // Test 1D tensor
        val tensor1d = CpuTensorFP32.zeros(Shape(5))
        val str1d = tensor1d.toString()
        println("[DEBUG_LOG] 1D FP32 Tensor toString: $str1d")
        assertTrue(str1d.contains("CpuTensorFP32"))
        assertTrue(str1d.contains("Shape: Dimensions = [5]"))
        assertTrue(str1d.contains("Size (Volume) = 5"))

        // Test 2D tensor
        val tensor2d = CpuTensorFP32.zeros(Shape(3, 4))
        val str2d = tensor2d.toString()
        println("[DEBUG_LOG] 2D FP32 Tensor toString: $str2d")
        assertTrue(str2d.contains("CpuTensorFP32"))
        assertTrue(str2d.contains("Shape: Dimensions = [3 x 4]"))
        assertTrue(str2d.contains("Size (Volume) = 12"))

        // Test 3D tensor
        val tensor3d = CpuTensorFP32.zeros(Shape(2, 3, 4))
        val str3d = tensor3d.toString()
        println("[DEBUG_LOG] 3D FP32 Tensor toString: $str3d")
        assertTrue(str3d.contains("CpuTensorFP32"))
        assertTrue(str3d.contains("Shape: Dimensions = [2 x 3 x 4]"))
        assertTrue(str3d.contains("Size (Volume) = 24"))
    }

    @Test
    fun testCpuTensorInt8ToString() {
        // Test 2D tensor
        val tensor2d = CpuTensorInt8.zeros(Shape(2, 3))
        val str2d = tensor2d.toString()
        println("[DEBUG_LOG] 2D Int8 Tensor toString: $str2d")
        assertTrue(str2d.contains("CpuTensorInt8"))
        assertTrue(str2d.contains("Shape: Dimensions = [2 x 3]"))
        assertTrue(str2d.contains("Size (Volume) = 6"))
    }

    @Test
    fun testCpuTensorInt32ToString() {
        // Test 2D tensor
        val tensor2d = CpuTensorInt32.zeros(Shape(4, 2))
        val str2d = tensor2d.toString()
        println("[DEBUG_LOG] 2D Int32 Tensor toString: $str2d")
        assertTrue(str2d.contains("CpuTensorInt32"))
        assertTrue(str2d.contains("Shape: Dimensions = [4 x 2]"))
        assertTrue(str2d.contains("Size (Volume) = 8"))
    }

    @Test
    fun testToStringForDebugging() {
        println("[DEBUG_LOG] Testing toString for debugging scenario...")
        
        // Create different tensors and print them for debugging
        val vector = CpuTensorFP32.ones(Shape(10))
        val matrix = CpuTensorFP32.zeros(Shape(5, 3))
        val tensor3d = CpuTensorInt8.zeros(Shape(2, 2, 2))
        
        println("[DEBUG_LOG] Vector: $vector")
        println("[DEBUG_LOG] Matrix: $matrix")  
        println("[DEBUG_LOG] 3D Tensor: $tensor3d")
        
        // These should clearly show the shapes for debugging
        assertTrue(vector.toString().contains("Shape: Dimensions = [10]"))
        assertTrue(matrix.toString().contains("Shape: Dimensions = [5 x 3]"))
        assertTrue(tensor3d.toString().contains("Shape: Dimensions = [2 x 2 x 2]"))
    }
}