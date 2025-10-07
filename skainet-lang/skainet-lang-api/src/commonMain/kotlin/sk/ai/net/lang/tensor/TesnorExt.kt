package sk.ai.net.lang.tensor

import sk.ai.net.lang.types.DType

public fun <T : DType, V> Tensor<T, V>.isScalar(): Boolean = this.shape.rank == 0
public fun <T : DType, V> Tensor<T, V>.isVector(): Boolean = this.shape.rank == 1
public fun <T : DType, V> Tensor<T, V>.isMatrix(): Boolean = this.shape.rank == 2