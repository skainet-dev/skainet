package sk.ai.net.impl

import sk.ai.net.DataDescriptor

class BuiltInDoubleDataDescriptor : DataDescriptor {
    override val name: String
        get() = "Double"
    override val bits: Int
        get() = Double.SIZE_BITS

    override fun isSigned(): Boolean = true

    override val minValue: Double = Double.MIN_VALUE

    override val maxValue: Double = Double.MAX_VALUE
}