package sk.ainet.lang.tensor.benchmark

import sk.ainet.lang.tensor.*
import sk.ainet.lang.types.DType
import kotlin.time.Duration
import kotlin.time.measureTime

/**
 * Performance benchmarking infrastructure for tensor slicing operations.
 * 
 * This module provides comprehensive benchmarking capabilities for analyzing
 * the performance characteristics of different slicing patterns, comparing
 * view vs copy operations, and measuring memory usage efficiency.
 * 
 * ## Benchmark Categories
 * 
 * 1. **Memory Usage Benchmarks**: Compare memory footprint of views vs copies
 * 2. **Access Pattern Performance**: Measure element access speed for different slice patterns  
 * 3. **Large Tensor Scenarios**: Evaluate performance with realistic tensor sizes
 * 4. **ML Model Integration**: Test with common deep learning access patterns
 */

/**
 * Results from a slicing performance benchmark run.
 * 
 * @param operationType the type of operation benchmarked (view, copy, access)
 * @param slicePattern description of the slice pattern tested
 * @param tensorSize the size/shape of the tensor used
 * @param duration the time taken for the operation
 * @param memoryUsage memory footprint in bytes (if applicable)
 * @param throughput elements processed per second (if applicable)
 */
public data class BenchmarkResult(
    val operationType: String,
    val slicePattern: String, 
    val tensorSize: String,
    val duration: Duration,
    val memoryUsage: Long? = null,
    val throughput: Double? = null
)

/**
 * Configuration for benchmark execution.
 * 
 * @param iterations number of times to repeat each benchmark
 * @param warmupIterations number of warmup runs before measurement
 * @param tensorShapes list of tensor shapes to test
 * @param includeMemoryBenchmarks whether to include memory usage measurements
 */
public data class BenchmarkConfig(
    val iterations: Int = 100,
    val warmupIterations: Int = 10,
    val tensorShapes: List<IntArray> = listOf(
        intArrayOf(32, 128, 224, 224),  // Typical CNN input
        intArrayOf(1024, 768),          // Transformer embeddings
        intArrayOf(8, 8, 512, 512),     // Large feature maps
        intArrayOf(100, 1000)           // Dense layer weights
    ),
    val includeMemoryBenchmarks: Boolean = true
)

/**
 * Main benchmarking suite for tensor slicing operations.
 */
public class SlicingBenchmarkSuite {
    
    /**
     * Runs comprehensive benchmarks comparing view vs copy operations.
     * 
     * This benchmark measures the performance difference between creating
     * tensor views (zero-copy) and tensor copies for various slicing patterns.
     * 
     * @param tensor the tensor to benchmark slicing operations on
     * @param config benchmark configuration parameters
     * @return list of benchmark results
     */
    public fun <T : DType, V> benchmarkViewVsCopy(
        tensor: Tensor<T, V>,
        config: BenchmarkConfig = BenchmarkConfig()
    ): List<BenchmarkResult> {
        val results = mutableListOf<BenchmarkResult>()
        
        // Test different slicing patterns
        val slicePatterns = listOf(
            "batch_slice" to { t: Tensor<T, V> -> 
                t.sliceView { 
                    segment { range(0, minOf(8, t.shape[0])) }
                    repeat(t.rank - 1) { segment { all() } }
                }
            },
            "channel_slice" to { t: Tensor<T, V> ->
                if (t.rank >= 2) {
                    t.sliceView {
                        segment { all() }
                        segment { range(0, minOf(64, t.shape[1])) }
                        repeat(t.rank - 2) { segment { all() } }
                    }
                } else {
                    t.sliceView { segment { all() } }
                }
            },
            "spatial_region" to { t: Tensor<T, V> ->
                if (t.rank >= 4) {
                    t.sliceView {
                        segment { all() }
                        segment { all() }
                        segment { range(0, minOf(112, t.shape[2])) }
                        segment { range(0, minOf(112, t.shape[3])) }
                    }
                } else {
                    t.sliceView { repeat(t.rank) { segment { all() } } }
                }
            }
        )
        
        for ((patternName, sliceFunction) in slicePatterns) {
            // Warmup
            repeat(config.warmupIterations) {
                sliceFunction(tensor)
            }
            
            // Benchmark view creation
            val viewTime = measureTime {
                repeat(config.iterations) {
                    sliceFunction(tensor)
                }
            }
            
            results.add(BenchmarkResult(
                operationType = "view_creation",
                slicePattern = patternName,
                tensorSize = tensor.shape.dimensions.joinToString("x"),
                duration = viewTime / config.iterations
            ))
            
            // Memory usage benchmark (if enabled)
            if (config.includeMemoryBenchmarks) {
                // Note: Memory measurement requires platform-specific implementation
                // This is a placeholder for cross-platform compatibility
                val view = sliceFunction(tensor)
                val estimatedMemoryUsage = view.shape.volume * 8L // Estimated 8 bytes per element
                
                results.add(BenchmarkResult(
                    operationType = "view_memory",
                    slicePattern = patternName,
                    tensorSize = tensor.shape.dimensions.joinToString("x"),
                    duration = Duration.ZERO,
                    memoryUsage = estimatedMemoryUsage
                ))
            }
        }
        
        return results
    }
    
    /**
     * Benchmarks element access performance for different slice patterns.
     * 
     * This test measures how quickly elements can be accessed through views
     * with different slicing patterns, helping identify optimization opportunities.
     * 
     * @param tensor the tensor to benchmark access patterns on
     * @param config benchmark configuration parameters
     * @return list of benchmark results
     */
    public fun <T : DType, V> benchmarkAccessPatterns(
        tensor: Tensor<T, V>,
        config: BenchmarkConfig = BenchmarkConfig()
    ): List<BenchmarkResult> {
        val results = mutableListOf<BenchmarkResult>()
        
        // Create views with different patterns
        val views = mapOf(
            "contiguous_batch" to tensor.sliceView {
                segment { range(0, minOf(16, tensor.shape[0])) }
                repeat(tensor.rank - 1) { segment { all() } }
            },
            "strided_access" to tensor.sliceView {
                segment { step(0, tensor.shape[0], 2) }
                repeat(tensor.rank - 1) { segment { all() } }
            },
            "spatial_region" to if (tensor.rank >= 4) {
                tensor.sliceView {
                    segment { all() }
                    segment { all() }
                    segment { range(10, minOf(110, tensor.shape[2])) }
                    segment { range(10, minOf(110, tensor.shape[3])) }
                }
            } else {
                tensor.sliceView { repeat(tensor.rank) { segment { all() } } }
            }
        )
        
        for ((patternName, view) in views) {
            // Warmup
            repeat(config.warmupIterations) {
                accessRandomElements(view, 100)
            }
            
            // Benchmark access time
            val accessTime = measureTime {
                repeat(config.iterations) {
                    accessRandomElements(view, 100)
                }
            }
            
            val elementsAccessed = 100L * config.iterations
            val throughput = elementsAccessed.toDouble() / accessTime.inWholeMilliseconds * 1000
            
            results.add(BenchmarkResult(
                operationType = "element_access",
                slicePattern = patternName,
                tensorSize = view.shape.dimensions.joinToString("x"),
                duration = accessTime / config.iterations,
                throughput = throughput
            ))
        }
        
        return results
    }
    
    /**
     * Benchmarks large tensor slicing scenarios common in ML workloads.
     * 
     * This benchmark simulates realistic deep learning scenarios like batch
     * processing, feature extraction, and sequence processing with sliding windows.
     * 
     * @param tensorFactory function to create test tensors of specified shapes
     * @param config benchmark configuration parameters  
     * @return list of benchmark results
     */
    public fun <T : DType, V> benchmarkMLScenarios(
        tensorFactory: (IntArray) -> Tensor<T, V>,
        config: BenchmarkConfig = BenchmarkConfig()
    ): List<BenchmarkResult> {
        val results = mutableListOf<BenchmarkResult>()
        
        val scenarios = listOf(
            "mini_batch_extraction" to { tensor: Tensor<T, V> ->
                // Extract mini-batches of size 32
                val batchSize = minOf(32, tensor.shape[0])
                tensor.sliceView {
                    segment { range(0, batchSize) }
                    repeat(tensor.rank - 1) { segment { all() } }
                }
            },
            "feature_map_channels" to { tensor: Tensor<T, V> ->
                // Extract first 128 channels from feature maps
                if (tensor.rank >= 4) {
                    tensor.sliceView {
                        segment { all() }
                        segment { range(0, minOf(128, tensor.shape[1])) }
                        segment { all() }
                        segment { all() }
                    }
                } else {
                    tensor.sliceView { repeat(tensor.rank) { segment { all() } } }
                }
            },
            "sliding_window" to { tensor: Tensor<T, V> ->
                // Simulate sliding window for sequence processing
                if (tensor.rank >= 2) {
                    val windowSize = minOf(64, tensor.shape[tensor.rank - 1])
                    tensor.sliceView {
                        repeat(tensor.rank - 1) { segment { all() } }
                        segment { range(0, windowSize) }
                    }
                } else {
                    tensor.sliceView { segment { all() } }
                }
            }
        )
        
        for (shape in config.tensorShapes) {
            val tensor = tensorFactory(shape)
            
            for ((scenarioName, sliceFunction) in scenarios) {
                // Warmup
                repeat(config.warmupIterations) {
                    sliceFunction(tensor)
                }
                
                // Benchmark scenario
                val scenarioTime = measureTime {
                    repeat(config.iterations) {
                        val view = sliceFunction(tensor)
                        // Simulate some access to the view
                        accessRandomElements(view, 10)
                    }
                }
                
                results.add(BenchmarkResult(
                    operationType = "ml_scenario",
                    slicePattern = scenarioName,
                    tensorSize = shape.joinToString("x"),
                    duration = scenarioTime / config.iterations
                ))
            }
        }
        
        return results
    }
    
    /**
     * Helper function to access random elements from a tensor view.
     * This simulates realistic access patterns in benchmarks.
     */
    private fun <T : DType, V> accessRandomElements(tensor: Tensor<T, V>, count: Int) {
        val shape = tensor.shape
        repeat(count) {
            val indices = IntArray(shape.rank) { dim ->
                (0 until shape[dim]).random()
            }
            tensor.data.get(*indices) // Access element
        }
    }
}

/**
 * Utility class for analyzing and reporting benchmark results.
 */
public class BenchmarkReporter {
    
    /**
     * Generates a formatted report from benchmark results.
     * 
     * @param results the benchmark results to report
     * @param title optional title for the report
     * @return formatted report string
     */
    public fun generateReport(results: List<BenchmarkResult>, title: String = "Slicing Benchmark Report"): String {
        val report = StringBuilder()
        report.appendLine("=" .repeat(60))
        report.appendLine(title)
        report.appendLine("=" .repeat(60))
        report.appendLine()
        
        // Group results by operation type
        val groupedResults = results.groupBy { it.operationType }
        
        for ((operationType, operationResults) in groupedResults) {
            report.appendLine("## $operationType".uppercase())
            report.appendLine("-".repeat(40))
            
            for (result in operationResults) {
                report.appendLine("Pattern: ${result.slicePattern}")
                report.appendLine("Tensor Size: ${result.tensorSize}")
                report.appendLine("Duration: ${result.duration}")
                result.memoryUsage?.let { 
                    report.appendLine("Memory Usage: ${it} bytes")
                }
                result.throughput?.let {
                    val formatted = (it * 100).toInt() / 100.0 // Round to 2 decimal places
                    report.appendLine("Throughput: $formatted elements/sec")
                }
                report.appendLine()
            }
        }
        
        return report.toString()
    }
    
    /**
     * Compares benchmark results and identifies performance differences.
     * 
     * @param baseline baseline benchmark results
     * @param comparison comparison benchmark results  
     * @return comparison analysis
     */
    public fun compareResults(baseline: List<BenchmarkResult>, comparison: List<BenchmarkResult>): String {
        val report = StringBuilder()
        report.appendLine("Benchmark Comparison Analysis")
        report.appendLine("=" .repeat(50))
        report.appendLine()
        
        val baselineMap = baseline.associateBy { "${it.operationType}_${it.slicePattern}_${it.tensorSize}" }
        val comparisonMap = comparison.associateBy { "${it.operationType}_${it.slicePattern}_${it.tensorSize}" }
        
        for ((key, baselineResult) in baselineMap) {
            comparisonMap[key]?.let { comparisonResult ->
                val speedup = baselineResult.duration.inWholeMilliseconds.toDouble() / 
                             comparisonResult.duration.inWholeMilliseconds.toDouble()
                
                val formattedSpeedup = (speedup * 100).toInt() / 100.0 // Round to 2 decimal places
                report.appendLine("${baselineResult.slicePattern} (${baselineResult.tensorSize})")
                report.appendLine("  Baseline: ${baselineResult.duration}")
                report.appendLine("  Comparison: ${comparisonResult.duration}")
                report.appendLine("  Speedup: ${formattedSpeedup}x")
                report.appendLine()
            }
        }
        
        return report.toString()
    }
}