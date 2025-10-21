package sk.ainet.lang.graph

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Execution modes supported by the framework
 */
public enum class ExecutionMode {
    /**
     * Eager execution - operations are executed immediately
     */
    EAGER,
    
    /**
     * Graph execution - operations are recorded and executed later
     */
    GRAPH
}

/**
 * Device types for tensor operations
 */
public enum class DeviceType {
    CPU,
    GPU,
    TPU,
    CUSTOM
}

/**
 * Device specification
 */
public data class Device(
    public val type: DeviceType,
    public val id: Int = 0,
    public val name: String = "${type.name.lowercase()}_$id"
)

/**
 * Context for managing execution state, including mode switching,
 * device management, and memory management.
 */
public interface ExecutionContext {
    
    /**
     * Current execution mode
     */
    public val executionMode: ExecutionMode
    
    /**
     * Current device for tensor operations
     */
    public val device: Device
    
    /**
     * Current execution tape (null if not recording)
     */
    public val currentTape: ExecutionTape?
    
    /**
     * Tape stack for nested execution contexts
     */
    public val tapeStack: TapeStack
    
    /**
     * Whether operations should be recorded
     */
    public val isRecording: Boolean get() = currentTape?.isRecording == true
    
    /**
     * Switch to eager execution mode
     */
    public fun switchToEager()
    
    /**
     * Switch to graph execution mode
     */
    public fun switchToGraph()
    
    /**
     * Set the device for tensor operations
     */
    public fun setDevice(device: Device)
    
    /**
     * Start recording operations with a new tape
     */
    public fun startRecording(tape: ExecutionTape = createTape())
    
    /**
     * Stop recording operations
     */
    public fun stopRecording(): ExecutionTape?
    
    /**
     * Execute an operation in the current context
     */
    public fun <T : DType, V> executeOperation(
        operation: Operation,
        inputs: List<Tensor<T, V>>
    ): List<Tensor<T, V>>
    
    /**
     * Create a new execution tape
     */
    public fun createTape(): ExecutionTape
    
    /**
     * Create a new gradient tape
     */
    public fun createGradientTape(): GradientTape
    
    /**
     * Execute with a specific execution mode temporarily
     */
    public fun <R> withExecutionMode(mode: ExecutionMode, block: () -> R): R
    
    /**
     * Execute with a specific device temporarily
     */
    public fun <R> withDevice(device: Device, block: () -> R): R
    
    /**
     * Execute with recording enabled temporarily
     */
    public fun <R> withRecording(tape: ExecutionTape = createTape(), block: () -> R): Pair<R, ExecutionTape>
    
    /**
     * Get memory usage information
     */
    public fun getMemoryInfo(): MemoryInfo
    
    /**
     * Force garbage collection
     */
    public fun collectGarbage()
    
    /**
     * Get execution statistics
     */
    public fun getExecutionStats(): ExecutionStats
    
    /**
     * Reset execution statistics
     */
    public fun resetExecutionStats()
}

/**
 * Memory usage information
 */
public data class MemoryInfo(
    /**
     * Total memory available on the device
     */
    public val totalMemory: Long,
    
    /**
     * Memory currently in use
     */
    public val usedMemory: Long,
    
    /**
     * Free memory available
     */
    public val freeMemory: Long = totalMemory - usedMemory,
    
    /**
     * Memory usage as percentage
     */
    public val usagePercentage: Double = (usedMemory.toDouble() / totalMemory) * 100.0
)

/**
 * Execution statistics
 */
public data class ExecutionStats(
    /**
     * Total number of operations executed
     */
    public val operationsExecuted: Long = 0,
    
    /**
     * Total execution time in milliseconds
     */
    public val totalExecutionTime: Long = 0,
    
    /**
     * Average execution time per operation
     */
    public val averageExecutionTime: Double = 
        if (operationsExecuted > 0) totalExecutionTime.toDouble() / operationsExecuted else 0.0,
    
    /**
     * Number of tensors created
     */
    public val tensorsCreated: Long = 0,
    
    /**
     * Peak memory usage
     */
    public val peakMemoryUsage: Long = 0
)

/**
 * Global execution context
 */
public object GlobalExecutionContext {
    private var _current: ExecutionContext? = null
    
    /**
     * Get the current global execution context
     */
    public fun current(): ExecutionContext {
        return _current ?: throw IllegalStateException("No execution context set")
    }
    
    /**
     * Set the global execution context
     */
    public fun setCurrent(context: ExecutionContext) {
        _current = context
    }
    
    /**
     * Check if a global execution context is set
     */
    public fun isSet(): Boolean = _current != null
}