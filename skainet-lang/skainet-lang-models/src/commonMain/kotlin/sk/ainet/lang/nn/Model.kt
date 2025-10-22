package sk.ainet.lang.nn

import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32

public interface Model {
    // output
    public fun <T : DType, V> model(): Module<FP32, Float>
    public fun modelCard(): String
}