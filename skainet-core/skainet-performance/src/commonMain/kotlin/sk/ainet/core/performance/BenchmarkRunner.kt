package sk.ainet.core.performance

import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds
import kotlin.time.measureTime
import kotlin.time.TimeSource

/**
 * Core benchmark runner for measuring tensor operation performance.
 * Multiplatform-compatible implementation using Kotlin stdlib APIs.
 */
public class BenchmarkRunner {
    
    /**
     * Runs a comprehensive benchmark of the given operation.
     * 
     * @param name Human-readable name for the benchmark
     * @param warmupRuns Number of warmup iterations (default: 10)
     * @param measurementRuns Number of measurement iterations (default: 100)
     * @param operation The operation to benchmark
     * @return Detailed benchmark results including timing statistics
     */
    public fun <T> benchmark(
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
    public fun warmupWithDuration(
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