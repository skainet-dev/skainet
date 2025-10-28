package sk.ainet.context

import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps

public interface ExecutionContext<V> {
    public val ops: TensorOps<V>
    public val tensorDataFactory: TensorDataFactory


    // runtime information
    public val memoryInfo:MemoryInfo
    public val executionStats:ExecutionStats
}
