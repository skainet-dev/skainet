package sk.ainet.lang.types

public object Int4 : DType {
    override val sizeInBits: Int = 4 // technically 4 bits, but minimum is 1 byte
    override val name: String = "Int4"
    override fun isCompatible(other: DType): Boolean {
        return when (other) {
            is Int8 -> true     // Same type compatibility
            is Ternary -> true     // Ternary + Int4 → Int8
            is FP32 -> true     // Can promote to FP32
            else -> false       // No other types supported yet
        }
    }

    override fun promoteTo(other: DType): DType {
        return when (other) {
            is Int8 -> Int8     // Int4 + Int8 → Int8
            is FP32 -> FP32     // Int4 + FP32 → FP32 (Int4 promotes to FP32)
            else -> throw IllegalArgumentException(
                "Cannot promote Int4 with incompatible type: ${other.name}"
            )
        }
    }
}