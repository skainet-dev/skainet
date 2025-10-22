package sk.ainet.lang.graph

import sk.ainet.context.ExecutionContext
import sk.ainet.context.ExecutionStats
import sk.ainet.context.MemoryInfo
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType


/**
 * Context for managing execution state, including mode switching,
 * device management, and memory management.
 */
public interface GraphExecutionContext<V> : ExecutionContext<V> {


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
