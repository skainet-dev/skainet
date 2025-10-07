package sk.ainet.lang.types

import kotlin.reflect.KClass

public fun DType.kotlinClass(): KClass<*> = when (this) {
    FP32 -> Float::class
    Int8 -> Byte::class
    Int4 -> Byte::class // best you can do, no 4-bit type
    Ternary -> Boolean::class // or custom
    FP16 -> Float::class
    Int32 -> Int::class
}