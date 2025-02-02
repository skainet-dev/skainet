package sk.ai.net.gguf.utils

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.reflect.KClass


enum class Endian {
    BIG_ENDIAN, LITTLE_ENDIAN
}

@OptIn(ExperimentalUnsignedTypes::class)
fun UByteArray.toUInt(endian: Endian): UInt {
    require(size >= 4) { "ByteArray size must be at least 4 to convert to UInt" }
    return when (endian) {
        Endian.BIG_ENDIAN -> ((this[0].toUInt() and 0xFFu) shl 24) or
                ((this[1].toUInt() and 0xFFu) shl 16) or
                ((this[2].toUInt() and 0xFFu) shl 8) or
                (this[3].toUInt() and 0xFFu)

        Endian.LITTLE_ENDIAN -> ((this[3].toUInt() and 0xFFu) shl 24) or
                ((this[2].toUInt() and 0xFFu) shl 16) or
                ((this[1].toUInt() and 0xFFu) shl 8) or
                (this[0].toUInt() and 0xFFu)
    }
}

@OptIn(ExperimentalUnsignedTypes::class)
fun UByteArray.toULong(endian: Endian): ULong {
    require(size >= 8) { "ByteArray size must be at least 8 to convert to ULong" }

    return when (endian) {
        Endian.BIG_ENDIAN -> ((this[0].toULong() and 0xFFuL) shl 56) or
                ((this[1].toULong() and 0xFFuL) shl 48) or
                ((this[2].toULong() and 0xFFuL) shl 40) or
                ((this[3].toULong() and 0xFFuL) shl 32) or
                ((this[4].toULong() and 0xFFuL) shl 24) or
                ((this[5].toULong() and 0xFFuL) shl 16) or
                ((this[6].toULong() and 0xFFuL) shl 8) or
                (this[7].toULong() and 0xFFuL)

        Endian.LITTLE_ENDIAN -> ((this[7].toULong() and 0xFFuL) shl 56) or
                ((this[6].toULong() and 0xFFuL) shl 48) or
                ((this[5].toULong() and 0xFFuL) shl 40) or
                ((this[4].toULong() and 0xFFuL) shl 32) or
                ((this[3].toULong() and 0xFFuL) shl 24) or
                ((this[2].toULong() and 0xFFuL) shl 16) or
                ((this[1].toULong() and 0xFFuL) shl 8) or
                (this[0].toULong() and 0xFFuL)
    }
}

@OptIn(ExperimentalUnsignedTypes::class)
fun UByteArray.toInt(endian: Endian): Int {
    require(size >= 4) { "ByteArray size must be at least 4 to convert to Int" }

    return when (endian) {
        Endian.BIG_ENDIAN -> ((this[0].toInt() and 0xFF) shl 24) or
                ((this[1].toInt() and 0xFF) shl 16) or
                ((this[2].toInt() and 0xFF) shl 8) or
                (this[3].toInt() and 0xFF)

        Endian.LITTLE_ENDIAN -> ((this[3].toInt() and 0xFF) shl 24) or
                ((this[2].toInt() and 0xFF) shl 16) or
                ((this[1].toInt() and 0xFF) shl 8) or
                (this[0].toInt() and 0xFF)
    }
}

//TODO add support for more data type (see GGUF support data types)
inline fun <reified T> ByteBuffer.readDataByType(
    offset: Int,
    dataCount: Int = 1,
    endian: Endian = Endian.LITTLE_ENDIAN,
): List<T> {
    val bytes = getSizeInByte(T::class)
    val length = bytes * dataCount

    val buffer = this.duplicate().apply {
        position(offset)
        limit(offset + length)
    }

    val destinationArray = ByteArray(length).also {
        buffer.get(it)
    }.map {
        it.toUByte()
    }

    return when (T::class) {
        UByte::class -> destinationArray.map { it as T }
        UInt::class -> destinationArray.chunked(4).map { it.toUByteArray().toUInt(endian) as T }
        ULong::class -> destinationArray.chunked(8)
            .map { it.toUByteArray().toULong(endian) as T }

        Int::class -> destinationArray.chunked(4).map { it.toUByteArray().toInt(endian) as T }
        Float::class -> destinationArray.chunked(4).map {
            ByteBuffer.wrap(it.toUByteArray().toByteArray())
                .order(if (endian == Endian.BIG_ENDIAN) ByteOrder.BIG_ENDIAN else ByteOrder.LITTLE_ENDIAN)
                .float as T
        }

        Boolean::class -> destinationArray.map { (it != 0.toUByte()) as T }
        else -> throw IllegalArgumentException("readDataByType: Unsupported type: ${T::class}")
    }
}

fun getSizeInByte(dataType: KClass<*>): Int {
    return when (dataType) {
        UByte::class -> 1
        Byte::class -> 1
        UShort::class -> 2
        Short::class -> 2
        UInt::class -> 4
        Int::class -> 4
        ULong::class -> 8
        Long::class -> 8
        Float::class -> 4
        Double::class -> 8
        Boolean::class -> 1
        else -> throw IllegalArgumentException("Unknown data type: $dataType")
    }
}

fun <E> List<E>.numberOfBytes(): Int {
    if (isEmpty()) {
        return 0
    }

    val elementSizeBytes = when (this[0]) {
        is UByte -> UByte.SIZE_BYTES
        is Byte -> Byte.SIZE_BYTES
        is UShort -> UShort.SIZE_BYTES
        is Short -> Short.SIZE_BYTES
        is UInt -> UInt.SIZE_BYTES
        is Int -> Int.SIZE_BYTES
        is Float -> Float.SIZE_BYTES
        is ULong -> ULong.SIZE_BYTES
        is Long -> Long.SIZE_BYTES
        is Double -> Double.SIZE_BYTES
        is Boolean -> 1 //TODO see if this is always true
        else -> throw IllegalArgumentException("Unsupported type: ${this[0]}")
    }
    return this.size * elementSizeBytes
}
