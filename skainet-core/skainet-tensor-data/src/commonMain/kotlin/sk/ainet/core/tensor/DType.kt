package sk.ainet.core.tensor

// Base marker interface for all dtypes
public sealed interface DType {
    public val sizeInBits: Int
    public val name: String
}

public object Ternary : DType {
    override val sizeInBits: Int = 2
    override val name: String = "Ternary"
}


public object Int4 : DType {
    override val sizeInBits: Int = 4 // technically 4 bits, but minimum is 1 byte
    override val name: String = "Int4"
}

public object Int8 : DType {
    override val sizeInBits: Int = 8 // technically 4 bits, but minimum is 1 byte
    override val name: String = "Int8"
}

public object Int32 : DType {
    override val sizeInBits: Int = 32
    override val name: String = "Int32"
}

public object FP16 : DType {
    override val sizeInBits: Int = 16
    override val name: String = "Float16"
}

public object FP32 : DType{
    override val sizeInBits: Int = 32
    override val name: String = "Float32"
}