package sk.ai.net.core.tensor.benchmark

import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds
import kotlin.time.measureTime
import kotlin.time.TimeSource

/**
 * Core benchmark runner for measuring tensor operation performance.
 * Multiplatform-compatible implementation using Kotlin stdlib APIs.
 */
class BenchmarkRunner {
    
    /**
     * Runs a comprehensive benchmark of the given operation.
     * 
     * @param name Human-readable name for the benchmark
     * @param warmupRuns Number of warmup iterations (default: 10)
     * @param measurementRuns Number of measurement iterations (default: 100)
     * @param operation The operation to benchmark
     * @return Detailed benchmark results including timing statistics
     */
    fun <T> benchmark(
        name: String,
        warmupRuns: Int = 10,
        measurementRuns: Int = 100,
        operation: () -> T
    ): BenchmarkResult {
        // Warmup phase to allow JIT compilation
        repeat(warmupRuns) {
            operation()
        }
        
        val times = mutableListOf<Double>()
        
        // Measurement phase
        repeat(measurementRuns) {
            val time = measureTime { operation() }.inWholeMicroseconds.toDouble()
            times.add(time)
        }
        
        return BenchmarkResult(
            name = name,
            executionTime = calculateTimeStatistics(times),
            memoryUsage = MemoryStatistics(0L, 0L, 0, 0L), // Placeholder for memory stats
            throughput = calculateThroughput(times),
            metadata = mapOf(
                "warmupRuns" to warmupRuns,
                "measurementRuns" to measurementRuns,
                "timestamp" to TimeSource.Monotonic.markNow().elapsedNow().inWholeMilliseconds
            )
        )
    }
    
    /**
     * Advanced warmup with time-based minimum duration
     */
    fun warmupWithDuration(
        operation: () -> Unit,
        minRuns: Int = 10,
        minTime: Duration = 5.seconds
    ) {
        val timeSource = TimeSource.Monotonic
        val startMark = timeSource.markNow()
        var runs = 0
        
        while (runs < minRuns || startMark.elapsedNow() < minTime) {
            operation()
            runs++
        }
    }
    
    private fun calculateTimeStatistics(times: List<Double>): TimeStatistics {
        val sorted = times.sorted()
        val mean = times.average()
        val variance = times.map { (it - mean) * (it - mean) }.average()
        val stdDev = kotlin.math.sqrt(variance)
        
        return TimeStatistics(
            mean = mean,
            median = sorted[sorted.size / 2],
            standardDeviation = stdDev,
            min = sorted.first(),
            max = sorted.last(),
            percentile95 = sorted[(sorted.size * 0.95).toInt()]
        )
    }
    
    private fun calculateThroughput(times: List<Double>): Double {
        val averageTimeMicros = times.average()
        return 1_000_000.0 / averageTimeMicros // operations per second
    }
}

/**
 * Comprehensive benchmark result containing timing statistics
 */
data class BenchmarkResult(
    val name: String,
    val executionTime: TimeStatistics,
    val memoryUsage: MemoryStatistics,
    val throughput: Double, // operations per second
    val metadata: Map<String, Any>
) {
    fun prettyPrint(): String {
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
data class TimeStatistics(
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
data class MemoryStatistics(
    val allocatedBytes: Long,
    val peakHeapUsage: Long,
    val gcCollections: Int,
    val gcTime: Long // milliseconds
)