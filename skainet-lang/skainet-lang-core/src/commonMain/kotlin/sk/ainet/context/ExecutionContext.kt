package sk.ainet.context

import sk.ainet.lang.tensor.ops.TensorOps


public interface ExecutionContext<V> {
    public val ops: TensorOps<V>
}
