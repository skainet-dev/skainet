package sk.ainet.core.tensor.backend

import sk.ainet.core.tensor.DType
import sk.ainet.core.tensor.TensorOps


/**
 * Interface representing a computation backend for tensor operations.
 *
 * A computation backend is responsible for executing tensor operations on a specific
 * hardware platform (CPU, GPU, etc.). Different backends can provide different
 * implementations of the same operations, optimized for their target platform.
 */
public interface ComputeBackend<D : DType, V> : TensorOps<D, V>> {
    /**
     * The name of the backend.
     */
    public val name: String


}