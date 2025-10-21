package sk.ainet.lang.graph

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Default implementation of ExecutionContext
 */
public class DefaultExecutionContext(
    initialMode: ExecutionMode = ExecutionMode.EAGER,
    initialDevice: Device = Device(DeviceType.CPU)
) : ExecutionContext {
    
    private var _executionMode: ExecutionMode = initialMode
    private var _device: Device = initialDevice
    private val _tapeStack: TapeStack = DefaultTapeStack()
    private var _executionStats: ExecutionStats = ExecutionStats()
    
    override val executionMode: ExecutionMode get() = _executionMode
    override val device: Device get() = _device
    override val currentTape: ExecutionTape? get() = _tapeStack.currentTape
    override val tapeStack: TapeStack get() = _tapeStack
    
    override fun switchToEager() {
        _executionMode = ExecutionMode.EAGER
    }
    
    override fun switchToGraph() {
        _executionMode = ExecutionMode.GRAPH
    }
    
    override fun setDevice(device: Device) {
        _device = device
    }
    
    override fun startRecording(tape: ExecutionTape) {
        _tapeStack.pushTape(tape)
        tape.startRecording()
    }
    
    override fun stopRecording(): ExecutionTape? {
        val tape = _tapeStack.popTape()
        tape?.stopRecording()
        return tape
    }
    
    override fun <T : DType, V> executeOperation(
        operation: Operation,
        inputs: List<Tensor<T, V>>
    ): List<Tensor<T, V>> {
        val startTime = getCurrentTime()
        
        try {
            // Execute the operation
            val outputs = operation.execute(inputs)
            
            // Record operation if tape is recording
            currentTape?.let { tape ->
                if (tape.isRecording) {
                    tape.recordOperation(operation, inputs, outputs)
                }
            }
            
            // Update execution statistics
            val executionTime = getCurrentTime() - startTime
            _executionStats = _executionStats.copy(
                operationsExecuted = _executionStats.operationsExecuted + 1,
                totalExecutionTime = _executionStats.totalExecutionTime + executionTime
            )
            
            return outputs
        } catch (e: Exception) {
            // Update error statistics if needed
            throw e
        }
    }
    
    override fun createTape(): ExecutionTape {
        return DefaultExecutionTape()
    }
    
    override fun createGradientTape(): GradientTape {
        return DefaultGradientTape()
    }
    
    override fun <R> withExecutionMode(mode: ExecutionMode, block: () -> R): R {
        val originalMode = _executionMode
        try {
            _executionMode = mode
            return block()
        } finally {
            _executionMode = originalMode
        }
    }
    
    override fun <R> withDevice(device: Device, block: () -> R): R {
        val originalDevice = _device
        try {
            _device = device
            return block()
        } finally {
            _device = originalDevice
        }
    }
    
    override fun <R> withRecording(tape: ExecutionTape, block: () -> R): Pair<R, ExecutionTape> {
        startRecording(tape)
        try {
            val result = block()
            return Pair(result, tape)
        } finally {
            stopRecording()
        }
    }
    
    override fun getMemoryInfo(): MemoryInfo {
        // TODO: Implement actual memory tracking
        // For now, return mock data
        return MemoryInfo(
            totalMemory = 1024 * 1024 * 1024, // 1GB mock
            usedMemory = 256 * 1024 * 1024    // 256MB mock
        )
    }
    
    override fun collectGarbage() {
        // TODO: Implement garbage collection if needed
        // For now, this is a no-op
    }
    
    override fun getExecutionStats(): ExecutionStats {
        return _executionStats
    }
    
    override fun resetExecutionStats() {
        _executionStats = ExecutionStats()
    }
    
    /**
     * Get current time for performance measurement
     * Using a simple counter since System.currentTimeMillis() is not available in common code
     */
    private fun getCurrentTime(): Long {
        // TODO: Implement proper time measurement for multiplatform
        // For now, return a simple counter
        return _executionStats.operationsExecuted
    }
}