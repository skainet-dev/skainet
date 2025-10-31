package sk.ainet.lang.types

/**
 * Custom data type holding 3 values -1,0,0 stored in 2 bits. Used e.g. with BitNet models
 * https://huggingface.co/microsoft/bitnet-b1.58-2B-4T
 */
public object Ternary : DType {
    override val sizeInBits: Int = 2
    override val name: String = "Ternary"
    override fun isCompatible(other: DType): Boolean {
        return when (other) {
            is Ternary -> true  // Same type compatibility
            is Int8 -> true     // Can promote to Int8
            is FP32 -> true     // Can promote to FP32
            else -> false       // No other types supported yet
        }
    }

    override fun promoteTo(other: DType): DType {
        return when (other) {
            is Ternary -> Ternary     // Ternary + Ternary → Ternary
            is Int8 -> Int8           // Ternary + Int8 → Int8
            is Int4 -> Int8           // Ternary + Int4 → Int8
            is FP32 -> FP32           // Ternary + FP32 → FP32 (Ternary promotes to FP32)
            else -> throw IllegalArgumentException(
                "Cannot promote Ternary with incompatible type: ${other.name}"
            )
        }
    }
}