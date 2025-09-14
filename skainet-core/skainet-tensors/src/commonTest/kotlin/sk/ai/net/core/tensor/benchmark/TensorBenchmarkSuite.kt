package sk.ai.net.core.tensor.benchmark

import sk.ai.net.core.tensor.*
import sk.ai.net.core.tensor.backend.*

/**
 * Comprehensive benchmark suite for tensor operations.
 * Tests performance of matrix multiplication, element-wise operations, and dot products
 * across different tensor sizes and data patterns.
 */
class TensorBenchmarkSuite(private val backend: CpuBackend) {
    
    private val runner = BenchmarkRunner()
    
    /**
     * Benchmark matrix multiplication across different sizes
     */
    fun benchmarkMatrixMultiplication(
        sizes: List<Triple<Int, Int, Int>> = defaultMatrixSizes
    ): List<BenchmarkResult> {
        return sizes.map { (rows, inner, cols) ->
            val a = createRandomTensor(Shape(rows, inner))
            val b = createRandomTensor(Shape(inner, cols))
            
            runner.benchmark(
                name = "MatMul ${rows}x${inner} * ${inner}x${cols}",
                warmupRuns = 5,
                measurementRuns = 50
            ) {
                backend.matmul(a, b)
            }
        }
    }
    
    /**
     * Benchmark element-wise operations (add, subtract, multiply, divide)
     */
    fun benchmarkElementwiseOps(
        sizes: List<Int> = defaultVectorSizes
    ): Map<String, List<BenchmarkResult>> {
        val operations = mapOf<String, (CpuTensorFP32, CpuTensorFP32) -> CpuTensorFP32>(
            "Add" to { a, b -> with(backend) { a + b } as CpuTensorFP32 },
            "Subtract" to { a, b -> with(backend) { a - b } as CpuTensorFP32 },
            "Multiply" to { a, b -> with(backend) { a * b } as CpuTensorFP32 },
            "Divide" to { a, b -> with(backend) { a / b } as CpuTensorFP32 }
        )
        
        return operations.mapValues { (opName, operation) ->
            sizes.map { size ->
                val a = createRandomTensor(Shape(size))
                val b = createRandomTensor(Shape(size))
                
                runner.benchmark(
                    name = "$opName (${size} elements)",
                    warmupRuns = 10,
                    measurementRuns = 100
                ) {
                    operation(a, b)
                }
            }
        }
    }
    
    /**
     * Benchmark dot product operations
     */
    fun benchmarkDotProduct(
        sizes: List<Int> = defaultVectorSizes
    ): List<BenchmarkResult> {
        return sizes.map { size ->
            val a = createRandomTensor(Shape(size))
            val b = createRandomTensor(Shape(size))
            
            runner.benchmark(
                name = "DotProduct (${size} elements)",
                warmupRuns = 10,
                measurementRuns = 100
            ) {
                backend.dot(a, b)
            }
        }
    }
    
    /**
     * Benchmark scalar operations
     */
    fun benchmarkScalarOps(
        sizes: List<Int> = defaultVectorSizes
    ): Map<String, List<BenchmarkResult>> {
        val scalarOps = mapOf<String, (CpuTensorFP32) -> CpuTensorFP32>(
            "Scale" to { tensor -> backend.scale(tensor, 2.5) as CpuTensorFP32 },
            "Add Scalar" to { tensor -> with(backend) { tensor + 3.14f } as CpuTensorFP32 },
            "Multiply Scalar" to { tensor -> with(backend) { tensor * 2.0f } as CpuTensorFP32 }
        )
        
        return scalarOps.mapValues { (opName, operation) ->
            sizes.map { size ->
                val tensor = createRandomTensor(Shape(size))
                
                runner.benchmark(
                    name = "$opName (${size} elements)",
                    warmupRuns = 10,
                    measurementRuns = 100
                ) {
                    operation(tensor)
                }
            }
        }
    }
    
    /**
     * Run comprehensive benchmark suite covering all operations
     */
    fun runFullBenchmarkSuite(): BenchmarkReport {
        println("Starting comprehensive tensor benchmark suite...")
        
        val matmulResults = benchmarkMatrixMultiplication()
        val elementwiseResults = benchmarkElementwiseOps()
        val dotProductResults = benchmarkDotProduct()
        val scalarResults = benchmarkScalarOps()
        
        return BenchmarkReport(
            backendName = backend.name,
            matrixMultiplication = matmulResults,
            elementwiseOperations = elementwiseResults,
            dotProduct = dotProductResults,
            scalarOperations = scalarResults,
            summary = generateSummary(matmulResults, elementwiseResults, dotProductResults, scalarResults)
        )
    }
    
    /**
     * Compare performance between two backends (e.g., scalar vs SIMD)
     */
    fun compareBackends(
        otherBackend: CpuBackend,
        otherBackendName: String
    ): ComparisonReport {
        println("Comparing performance between ${backend.name} and $otherBackendName...")
        
        val thisResults = runFullBenchmarkSuite()
        val otherSuite = TensorBenchmarkSuite(otherBackend)
        val otherResults = otherSuite.runFullBenchmarkSuite()
        
        return ComparisonReport(
            baselineBackend = backend.name,
            baselineResults = thisResults,
            comparisonBackend = otherBackendName,
            comparisonResults = otherResults,
            speedupAnalysis = calculateSpeedups(thisResults, otherResults)
        )
    }
    
    private fun createRandomTensor(shape: Shape): CpuTensorFP32 {
        val size = shape.volume
        val data = FloatArray(size) { (it % 100 + 1).toFloat() / 50f } // Values between 0.02 and 2.0
        return CpuTensorFP32.fromArray(shape, data)
    }
    
    private fun generateSummary(
        matmul: List<BenchmarkResult>,
        elementwise: Map<String, List<BenchmarkResult>>,
        dotProduct: List<BenchmarkResult>,
        scalar: Map<String, List<BenchmarkResult>>
    ): BenchmarkSummary {
        val allResults = buildList {
            addAll(matmul)
            elementwise.values.forEach { addAll(it) }
            addAll(dotProduct)
            scalar.values.forEach { addAll(it) }
        }
        
        return BenchmarkSummary(
            totalOperations = allResults.size,
            averageThroughput = allResults.map { it.throughput }.average(),
            fastestOperation = allResults.maxByOrNull { it.throughput }?.name ?: "N/A",
            slowestOperation = allResults.minByOrNull { it.throughput }?.name ?: "N/A"
        )
    }
    
    private fun calculateSpeedups(
        baseline: BenchmarkReport,
        comparison: BenchmarkReport
    ): SpeedupAnalysis {
        val speedups = mutableMapOf<String, Double>()
        
        // Calculate speedups for matrix multiplication
        baseline.matrixMultiplication.zip(comparison.matrixMultiplication) { base, comp ->
            speedups[base.name] = base.executionTime.mean / comp.executionTime.mean
        }
        
        // Calculate speedups for element-wise operations
        baseline.elementwiseOperations.forEach { (opType, baseResults) ->
            val compResults = comparison.elementwiseOperations[opType] ?: emptyList()
            baseResults.zip(compResults) { base, comp ->
                speedups[base.name] = base.executionTime.mean / comp.executionTime.mean
            }
        }
        
        val speedupValues = speedups.values.toList()
        
        return SpeedupAnalysis(
            operationSpeedups = speedups,
            averageSpeedup = speedupValues.average(),
            bestCaseSpeedup = speedupValues.maxOrNull() ?: 1.0,
            worstCaseSpeedup = speedupValues.minOrNull() ?: 1.0,
            speedupByTensorSize = speedups // Simplified - could be grouped by size
        )
    }
    
    companion object {
        // Default benchmark sizes
        val defaultMatrixSizes = listOf(
            Triple(32, 32, 32),      // Small
            Triple(64, 64, 64),      
            Triple(128, 128, 128),   // Medium
            Triple(256, 256, 256),
            Triple(512, 512, 512),   // Large
            Triple(1024, 1024, 1024),
            Triple(100, 500, 200),   // Rectangular
            Triple(1000, 10, 1000),  // Thin
            Triple(10, 1000, 10)     // Wide
        )
        
        val defaultVectorSizes = listOf(
            100, 1_000, 10_000, 100_000, 1_000_000
        )
    }
}

/**
 * Comprehensive benchmark report containing all operation results
 */
data class BenchmarkReport(
    val backendName: String,
    val matrixMultiplication: List<BenchmarkResult>,
    val elementwiseOperations: Map<String, List<BenchmarkResult>>,
    val dotProduct: List<BenchmarkResult>,
    val scalarOperations: Map<String, List<BenchmarkResult>>,
    val summary: BenchmarkSummary
) {
    fun prettyPrint(): String {
        return buildString {
            appendLine("=" .repeat(50))
            appendLine("TENSOR BENCHMARK REPORT - $backendName")
            appendLine("=" .repeat(50))
            appendLine()
            
            appendLine("MATRIX MULTIPLICATION:")
            matrixMultiplication.forEach { result ->
                appendLine(result.prettyPrint())
            }
            appendLine()
            
            appendLine("ELEMENT-WISE OPERATIONS:")
            elementwiseOperations.forEach { (opType, results) ->
                appendLine("  $opType:")
                results.forEach { result ->
                    appendLine("    ${result.prettyPrint().prependIndent("  ")}")
                }
            }
            appendLine()
            
            appendLine("SUMMARY:")
            appendLine("  Total Operations: ${summary.totalOperations}")
            appendLine("  Average Throughput: ${summary.averageThroughput.format(1)} ops/sec")
            appendLine("  Fastest: ${summary.fastestOperation}")
            appendLine("  Slowest: ${summary.slowestOperation}")
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

/**
 * Summary statistics for a benchmark run
 */
data class BenchmarkSummary(
    val totalOperations: Int,
    val averageThroughput: Double,
    val fastestOperation: String,
    val slowestOperation: String
)

/**
 * Backend comparison report showing speedup analysis
 */
data class ComparisonReport(
    val baselineBackend: String,
    val baselineResults: BenchmarkReport,
    val comparisonBackend: String,
    val comparisonResults: BenchmarkReport,
    val speedupAnalysis: SpeedupAnalysis
) {
    fun prettyPrint(): String {
        return buildString {
            appendLine("=" .repeat(60))
            appendLine("BACKEND COMPARISON: $baselineBackend vs $comparisonBackend")
            appendLine("=" .repeat(60))
            appendLine()
            
            appendLine("SPEEDUP ANALYSIS:")
            appendLine("  Average Speedup: ${speedupAnalysis.averageSpeedup.format(2)}x")
            appendLine("  Best Case: ${speedupAnalysis.bestCaseSpeedup.format(2)}x")
            appendLine("  Worst Case: ${speedupAnalysis.worstCaseSpeedup.format(2)}x")
            appendLine()
            
            appendLine("DETAILED SPEEDUPS:")
            speedupAnalysis.operationSpeedups.forEach { (operation, speedup) ->
                appendLine("  $operation: ${speedup.format(2)}x")
            }
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

/**
 * Speedup analysis comparing two backends
 */
data class SpeedupAnalysis(
    val operationSpeedups: Map<String, Double>,
    val averageSpeedup: Double,
    val bestCaseSpeedup: Double,
    val worstCaseSpeedup: Double,
    val speedupByTensorSize: Map<String, Double>
)