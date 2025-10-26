package sk.ainet.context

import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.sk.ainet.exec.tensor.ops.DefaultCpuOps

public class DirectCpuExecutionContext<V> : ExecutionContext<V> {
    override val tensorDataFactory: TensorDataFactory = DenseTensorDataFactory()

    override val ops: TensorOps<V>
        get() = DefaultCpuOps(tensorDataFactory)
}