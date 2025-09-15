package sk.ainet.core.performance

/**
 * Comprehensive benchmark result containing timing statistics
 */
public data class BenchmarkResult(
    val name: String,
    val executionTime: TimeStatistics,
    val memoryUsage: MemoryStatistics,
    val throughput: Double, // operations per second
    val metadata: Map<String, Any>
) {
    public fun prettyPrint(): String {
        return buildString {
            appendLine("$name:")
            appendLine("  Time: ${(executionTime.mean / 1000).format(3)}ms ± ${(executionTime.standardDeviation / 1000).format(3)}ms")
            appendLine("  Throughput: ${throughput.format(1)} ops/sec")
            appendLine("  Memory: ${formatBytes(memoryUsage.allocatedBytes)}")
            appendLine("  Range: ${(executionTime.min / 1000).format(3)}ms - ${(executionTime.max / 1000).format(3)}ms")
        }
    }
    
    private fun formatBytes(bytes: Long): String {
        return when {
            bytes >= 1_000_000 -> "${(bytes / 1_000_000.0).format(1)}MB"
            bytes >= 1_000 -> "${(bytes / 1_000.0).format(1)}KB"
            else -> "${bytes}B"
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
 * Statistical analysis of execution times
 */
public data class TimeStatistics(
    val mean: Double, // microseconds
    val median: Double, // microseconds
    val standardDeviation: Double, // microseconds
    val min: Double, // microseconds
    val max: Double, // microseconds
    val percentile95: Double // microseconds
)

/**
 * Memory usage statistics (placeholder for multiplatform compatibility)
 */
public data class MemoryStatistics(
    val allocatedBytes: Long,
    val peakHeapUsage: Long,
    val gcCollections: Int,
    val gcTime: Long // milliseconds
)

/**
 * Comprehensive benchmark report containing all operation results
 */
public data class BenchmarkReport(
    val backendName: String,
    val matrixMultiplication: List<BenchmarkResult>,
    val elementwiseOperations: Map<String, List<BenchmarkResult>>,
    val dotProduct: List<BenchmarkResult>,
    val scalarOperations: Map<String, List<BenchmarkResult>>,
    val summary: BenchmarkSummary
) {
    public fun prettyPrint(): String {
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
public data class BenchmarkSummary(
    val totalOperations: Int,
    val averageThroughput: Double,
    val fastestOperation: String,
    val slowestOperation: String
)

/**
 * Backend comparison report showing speedup analysis
 */
public data class ComparisonReport(
    val baselineBackend: String,
    val baselineResults: BenchmarkReport,
    val comparisonBackend: String,
    val comparisonResults: BenchmarkReport,
    val speedupAnalysis: SpeedupAnalysis
) {
    public fun prettyPrint(): String {
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
public data class SpeedupAnalysis(
    val operationSpeedups: Map<String, Double>,
    val averageSpeedup: Double,
    val bestCaseSpeedup: Double,
    val worstCaseSpeedup: Double,
    val speedupByTensorSize: Map<String, Double>
)