package sk.ai.net

/**
 * A simple interface to describe a data type that can be used in Tensors.
 */
interface DataDescriptor {
    /** Name of the data type (e.g."Int4", "Int8", "Float32", "Float4", "BFloat16"). */
    val name: String

    /** Number of bits used by the data type. */
    val bits: Int

    /** True if the data type is signed, false otherwise. */
    fun isSigned(): Boolean

    /** Minimul value that can be represented by the data type. */
    val minValue: Double

    /** Maximum value that can be represented by the data type. */
    val maxValue: Double

    /** True if value is a floating point */
    val isFloatingPoint: Boolean
}
