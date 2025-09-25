package sk.ainet.core.tensor.benchmark

import sk.ainet.core.performance.*
import kotlin.test.Ignore
import kotlin.test.Test

/**
 * Simple performance test demonstrating the extracted benchmarking framework.
 * This validates that the performance module integration works correctly.
 */
class BaselinePerformanceTest {
    
    private val runner = BenchmarkRunner()
    
    @Test
    @Ignore
    fun testBenchmarkRunnerIntegration() {
        println("\n=== BENCHMARK RUNNER INTEGRATION TEST ===")
        
        // Simple computation benchmark
        val result = runner.benchmark(
            name = "Simple Array Sum",
            warmupRuns = 3,
            measurementRuns = 10
        ) {
            val array = IntArray(1000) { it }
            array.sum()
        }
        
        println(result.prettyPrint())
        
        // Verify the benchmark produces valid results
        kotlin.test.assertTrue(result.throughput > 0, "Throughput should be positive")
        kotlin.test.assertTrue(result.executionTime.mean > 0, "Mean execution time should be positive")
        kotlin.test.assertEquals("Simple Array Sum", result.name)
        
        println("✓ Performance module integration successful")
    }
    
    @Test 
    fun testWarmupWithDuration() {
        println("\n=== WARMUP WITH DURATION TEST ===")
        
        var counter = 0
        runner.warmupWithDuration(
            operation = { counter++ },
            minRuns = 5,
            minTime = kotlin.time.Duration.parse("100ms")
        )
        
        kotlin.test.assertTrue(counter >= 5, "Should run at least minRuns times")
        println("✓ Warmup completed with $counter runs")
    }
}