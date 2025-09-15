package sk.ai.net.core.tensor.benchmark

import sk.ai.net.core.tensor.backend.CpuBackend
import kotlin.test.Test

/**
 * Baseline performance test demonstrating the benchmarking framework.
 * This test measures the current performance of CpuBackend operations
 * to establish baseline metrics for SIMD comparison.
 */
class BaselinePerformanceTest {
    
    private val backend = CpuBackend()
    private val suite = TensorBenchmarkSuite(backend)
    
    @Test
    fun testMatrixMultiplicationBaseline() {
        println("\n=== MATRIX MULTIPLICATION BASELINE ===")
        
        // Test smaller sizes for quick baseline measurement
        val testSizes = listOf(
            Triple(32, 32, 32),
            Triple(64, 64, 64),
            Triple(128, 128, 128),
            Triple(256, 256, 256)
        )
        
        val results = suite.benchmarkMatrixMultiplication(testSizes)
        
        results.forEach { result ->
            println(result.prettyPrint())
            println()
        }
        
        // Validate performance characteristics
        println("Performance Analysis:")
        results.forEach { result ->
            val size = extractMatrixSize(result.name)
            val operations = size * size * size.toLong() // Approximate FLOPs
            val flops = operations * result.throughput
            println("  ${result.name}: ${formatFlops(flops)}")
        }
    }
    
    @Test
    fun testElementwiseOperationsBaseline() {
        println("\n=== ELEMENT-WISE OPERATIONS BASELINE ===")
        
        // Test moderate sizes for element-wise operations
        val testSizes = listOf(1_000, 10_000, 100_000, 1_000_000)
        
        val results = suite.benchmarkElementwiseOps(testSizes)
        
        results.forEach { (operation, opResults) ->
            println("$operation Operations:")
            opResults.forEach { result ->
                println("  ${result.prettyPrint()}")
            }
            println()
        }
    }
    
    @Test
    fun testDotProductBaseline() {
        println("\n=== DOT PRODUCT BASELINE ===")
        
        val testSizes = listOf(1_000, 10_000, 100_000, 1_000_000)
        
        val results = suite.benchmarkDotProduct(testSizes)
        
        results.forEach { result ->
            println(result.prettyPrint())
            
            // Calculate memory bandwidth estimation
            val size = extractVectorSize(result.name)
            val bytesProcessed = size * 2 * 4L // Two arrays of floats (4 bytes each)
            val bandwidth = bytesProcessed * result.throughput / 1_000_000_000.0 // GB/s
            println("  Estimated Memory Bandwidth: ${bandwidth.format(2)} GB/s")
            println()
        }
    }
    
    @Test
    fun testScalarOperationsBaseline() {
        println("\n=== SCALAR OPERATIONS BASELINE ===")
        
        val testSizes = listOf(10_000, 100_000, 1_000_000)
        
        val results = suite.benchmarkScalarOps(testSizes)
        
        results.forEach { (operation, opResults) ->
            println("$operation Operations:")
            opResults.forEach { result ->
                println("  ${result.prettyPrint()}")
            }
            println()
        }
    }
    
    @Test
    fun testFullBenchmarkSuite() {
        println("\n=== COMPREHENSIVE BASELINE BENCHMARK ===")
        
        val report = suite.runFullBenchmarkSuite()
        println(report.prettyPrint())
        
        // Performance insights
        println("\nPERFORMACE INSIGHTS:")
        
        // Matrix multiplication scaling analysis
        val matmulResults = report.matrixMultiplication.take(4) // First 4 square matrices
        if (matmulResults.size >= 2) {
            val small = matmulResults[0]
            val large = matmulResults[3]
            val sizeRatio = 8.0 // 256/32 = 8
            val timeRatio = large.executionTime.mean / small.executionTime.mean
            val expectedRatio = sizeRatio * sizeRatio * sizeRatio // O(n³)
            
            println("  Matrix Multiplication Scaling:")
            println("    Size ratio: ${sizeRatio}x")
            println("    Time ratio: ${timeRatio.format(2)}x")
            println("    Expected O(n³) ratio: ${expectedRatio.format(1)}x")
            println("    Efficiency: ${(expectedRatio / timeRatio * 100).format(1)}%")
        }
        
        // Element-wise operations should scale linearly
        val addResults = report.elementwiseOperations["Add"] ?: emptyList()
        if (addResults.size >= 2) {
            val small = addResults[0]
            val large = addResults.last()
            val smallSize = extractVectorSize(small.name)
            val largeSize = extractVectorSize(large.name)
            val sizeRatio = largeSize.toDouble() / smallSize
            val timeRatio = large.executionTime.mean / small.executionTime.mean
            
            println("  Element-wise Addition Scaling:")
            println("    Size ratio: ${sizeRatio.format(1)}x")
            println("    Time ratio: ${timeRatio.format(2)}x")
            println("    Linearity: ${(sizeRatio / timeRatio * 100).format(1)}%")
        }
    }
    
    @Test
    fun testMemoryUsageDemonstration() {
        println("\n=== MEMORY USAGE ANALYSIS ===")
        
        // Note: Current implementation has placeholder memory stats
        // This test demonstrates where real memory profiling would go
        
        val runner = BenchmarkRunner()
        
        // Large matrix multiplication to observe memory patterns
        val result = runner.benchmark(
            name = "Large Matrix Memory Test",
            warmupRuns = 2,
            measurementRuns = 5
        ) {
            val backend = CpuBackend()
            val a = createLargeMatrix(500, 500)
            val b = createLargeMatrix(500, 500)
            backend.matmul(a, b)
        }
        
        println(result.prettyPrint())
        println("Note: Memory statistics are placeholders in current multiplatform implementation")
        println("Real JVM implementation would show:")
        println("  - Heap allocation patterns")
        println("  - GC pressure")
        println("  - Memory bandwidth utilization")
    }
    
    private fun createLargeMatrix(rows: Int, cols: Int) = 
        sk.ai.net.core.tensor.backend.CpuTensorFP32.fromArray(
            sk.ai.net.core.tensor.Shape(rows, cols),
            FloatArray(rows * cols) { (it % 100).toFloat() / 50f }
        )
    
    private fun extractMatrixSize(name: String): Int {
        // Extract size from "MatMul 256x256 * 256x256" format
        val regex = """MatMul (\d+)x\d+ \* \d+x\d+""".toRegex()
        val match = regex.find(name)
        return match?.groupValues?.get(1)?.toInt() ?: 32
    }
    
    private fun extractVectorSize(name: String): Int {
        // Extract size from "DotProduct (1000000 elements)" format
        val regex = """\((\d+) elements\)""".toRegex()
        val match = regex.find(name)
        return match?.groupValues?.get(1)?.toInt() ?: 1000
    }
    
    private fun formatFlops(flops: Double): String {
        return when {
            flops >= 1_000_000_000 -> "${(flops / 1_000_000_000).format(2)} GFLOP/s"
            flops >= 1_000_000 -> "${(flops / 1_000_000).format(2)} MFLOP/s"
            flops >= 1_000 -> "${(flops / 1_000).format(2)} KFLOP/s"
            else -> "${flops.format(2)} FLOP/s"
        }
    }
    
    private fun Double.format(decimals: Int): String {
        val multiplier = when (decimals) {
            0 -> 1.0
            1 -> 10.0
            2 -> 100.0
            3 -> 1000.0
            else -> 1000.0
        }
        val rounded = (this * multiplier).toInt() / multiplier
        return rounded.toString()
    }
}