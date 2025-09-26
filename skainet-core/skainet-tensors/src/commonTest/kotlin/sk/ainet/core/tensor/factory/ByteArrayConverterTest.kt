package sk.ainet.core.tensor.factory

import sk.ainet.core.tensor.*
import kotlin.test.*

/**
 * Comprehensive tests for ByteArrayConverter functionality.
 * Tests float conversion, int conversion, endianness handling, and validation.
 */
class ByteArrayConverterTest {

    @BeforeTest
    fun setup() {
        // Setup if needed
    }

    @AfterTest
    fun cleanup() {
        // Cleanup if needed
    }

    @Test
    fun testFloatConversionWithKnownPatterns() {
        // Test converting known byte patterns to floats
        
        // 1.0f in little-endian: 0x3F800000 -> [0x00, 0x00, 0x80, 0x3F]
        val oneFloatBytes = byteArrayOf(0x00, 0x00, 0x80.toByte(), 0x3F)
        val oneFloatResult = ByteArrayConverter.convertBytesToFloatArray(oneFloatBytes)
        assertEquals(1, oneFloatResult.size, "Should have 1 float")
        assertEquals(1.0f, oneFloatResult[0], 0.001f, "Should convert to 1.0f")
        
        // 2.0f in little-endian: 0x40000000 -> [0x00, 0x00, 0x00, 0x40]
        val twoFloatBytes = byteArrayOf(0x00, 0x00, 0x00, 0x40)
        val twoFloatResult = ByteArrayConverter.convertBytesToFloatArray(twoFloatBytes)
        assertEquals(1, twoFloatResult.size, "Should have 1 float")
        assertEquals(2.0f, twoFloatResult[0], 0.001f, "Should convert to 2.0f")
        
        // -1.0f in little-endian: 0xBF800000 -> [0x00, 0x00, 0x80, 0xBF]
        val negativeOneBytes = byteArrayOf(0x00, 0x00, 0x80.toByte(), 0xBF.toByte())
        val negativeOneResult = ByteArrayConverter.convertBytesToFloatArray(negativeOneBytes)
        assertEquals(1, negativeOneResult.size, "Should have 1 float")
        assertEquals(-1.0f, negativeOneResult[0], 0.001f, "Should convert to -1.0f")
    }

    @Test
    fun testIntConversionWithKnownPatterns() {
        // Test converting known byte patterns to ints
        
        // 100 in little-endian: 0x00000064 -> [0x64, 0x00, 0x00, 0x00]
        val hundredBytes = byteArrayOf(0x64, 0x00, 0x00, 0x00)
        val hundredResult = ByteArrayConverter.convertBytesToIntArray(hundredBytes)
        assertEquals(1, hundredResult.size, "Should have 1 int")
        assertEquals(100, hundredResult[0], "Should convert to 100")
        
        // -1 in little-endian: 0xFFFFFFFF -> [0xFF, 0xFF, 0xFF, 0xFF]
        val negativeOneBytes = byteArrayOf(0xFF.toByte(), 0xFF.toByte(), 0xFF.toByte(), 0xFF.toByte())
        val negativeOneResult = ByteArrayConverter.convertBytesToIntArray(negativeOneBytes)
        assertEquals(1, negativeOneResult.size, "Should have 1 int")
        assertEquals(-1, negativeOneResult[0], "Should convert to -1")
        
        // 65536 in little-endian: 0x00010000 -> [0x00, 0x00, 0x01, 0x00]
        val largeBytes = byteArrayOf(0x00, 0x00, 0x01, 0x00)
        val largeResult = ByteArrayConverter.convertBytesToIntArray(largeBytes)
        assertEquals(1, largeResult.size, "Should have 1 int")
        assertEquals(65536, largeResult[0], "Should convert to 65536")
    }

    @Test
    fun testByteConversionIdentity() {
        // Test byte array identity conversion
        val originalBytes = byteArrayOf(1, 2, 3, 4, 5, -1, -2, 127, -128)
        val convertedBytes = ByteArrayConverter.convertBytesToByteArray(originalBytes)
        
        assertEquals(originalBytes.size, convertedBytes.size, "Size should be preserved")
        assertContentEquals(originalBytes, convertedBytes, "Byte arrays should be identical")
        
        // Test empty array
        val emptyBytes = byteArrayOf()
        val emptyConverted = ByteArrayConverter.convertBytesToByteArray(emptyBytes)
        assertEquals(0, emptyConverted.size, "Empty array should remain empty")
        assertContentEquals(emptyBytes, emptyConverted, "Empty arrays should be identical")
    }

    @Test
    fun testMultipleFloatConversion() {
        // Test converting multiple floats at once
        val floats = floatArrayOf(1.0f, 2.5f, -3.7f, 0.0f)
        val bytes = ByteArray(16) // 4 floats * 4 bytes each
        
        // Manually encode floats to bytes (little-endian)
        for (i in floats.indices) {
            val bits = floats[i].toBits()
            bytes[i * 4 + 0] = (bits and 0xFF).toByte()
            bytes[i * 4 + 1] = ((bits shr 8) and 0xFF).toByte()
            bytes[i * 4 + 2] = ((bits shr 16) and 0xFF).toByte()
            bytes[i * 4 + 3] = ((bits shr 24) and 0xFF).toByte()
        }
        
        val converted = ByteArrayConverter.convertBytesToFloatArray(bytes)
        assertEquals(4, converted.size, "Should have 4 floats")
        
        for (i in floats.indices) {
            assertEquals(floats[i], converted[i], 0.001f, "Float $i should match")
        }
    }

    @Test
    fun testMultipleIntConversion() {
        // Test converting multiple ints at once
        val ints = intArrayOf(100, -200, 0, Int.MAX_VALUE, Int.MIN_VALUE)
        val bytes = ByteArray(20) // 5 ints * 4 bytes each
        
        // Manually encode ints to bytes (little-endian)
        for (i in ints.indices) {
            bytes[i * 4 + 0] = (ints[i] and 0xFF).toByte()
            bytes[i * 4 + 1] = ((ints[i] shr 8) and 0xFF).toByte()
            bytes[i * 4 + 2] = ((ints[i] shr 16) and 0xFF).toByte()
            bytes[i * 4 + 3] = ((ints[i] shr 24) and 0xFF).toByte()
        }
        
        val converted = ByteArrayConverter.convertBytesToIntArray(bytes)
        assertEquals(5, converted.size, "Should have 5 ints")
        
        for (i in ints.indices) {
            assertEquals(ints[i], converted[i], "Int $i should match")
        }
    }

    @Test
    fun testFloatValidationAlignment() {
        // Test that float conversion validates byte array size alignment
        
        // Valid sizes (multiples of 4)
        val validBytes4 = ByteArray(4)
        val validBytes8 = ByteArray(8)
        val validBytes12 = ByteArray(12)
        
        // These should not throw exceptions
        val result4 = ByteArrayConverter.convertBytesToFloatArray(validBytes4)
        assertEquals(1, result4.size, "4 bytes should produce 1 float")
        
        val result8 = ByteArrayConverter.convertBytesToFloatArray(validBytes8)
        assertEquals(2, result8.size, "8 bytes should produce 2 floats")
        
        val result12 = ByteArrayConverter.convertBytesToFloatArray(validBytes12)
        assertEquals(3, result12.size, "12 bytes should produce 3 floats")
        
        // Invalid sizes (not multiples of 4)
        val invalidBytes1 = ByteArray(1)
        val invalidBytes2 = ByteArray(2)
        val invalidBytes3 = ByteArray(3)
        val invalidBytes5 = ByteArray(5)
        
        assertFailsWith<IllegalArgumentException>("1 byte should be invalid") {
            ByteArrayConverter.convertBytesToFloatArray(invalidBytes1)
        }
        assertFailsWith<IllegalArgumentException>("2 bytes should be invalid") {
            ByteArrayConverter.convertBytesToFloatArray(invalidBytes2)
        }
        assertFailsWith<IllegalArgumentException>("3 bytes should be invalid") {
            ByteArrayConverter.convertBytesToFloatArray(invalidBytes3)
        }
        assertFailsWith<IllegalArgumentException>("5 bytes should be invalid") {
            ByteArrayConverter.convertBytesToFloatArray(invalidBytes5)
        }
    }

    @Test
    fun testIntValidationAlignment() {
        // Test that int conversion validates byte array size alignment
        
        // Valid sizes (multiples of 4)
        val validBytes4 = ByteArray(4)
        val validBytes8 = ByteArray(8)
        val validBytes16 = ByteArray(16)
        
        // These should not throw exceptions
        val intResult4 = ByteArrayConverter.convertBytesToIntArray(validBytes4)
        assertEquals(1, intResult4.size, "4 bytes should produce 1 int")
        
        val intResult8 = ByteArrayConverter.convertBytesToIntArray(validBytes8)
        assertEquals(2, intResult8.size, "8 bytes should produce 2 ints")
        
        val intResult16 = ByteArrayConverter.convertBytesToIntArray(validBytes16)
        assertEquals(4, intResult16.size, "16 bytes should produce 4 ints")
        
        // Invalid sizes (not multiples of 4)
        val invalidBytes1 = ByteArray(1)
        val invalidBytes2 = ByteArray(2)
        val invalidBytes3 = ByteArray(3)
        val invalidBytes7 = ByteArray(7)
        
        assertFailsWith<IllegalArgumentException>("1 byte should be invalid") {
            ByteArrayConverter.convertBytesToIntArray(invalidBytes1)
        }
        assertFailsWith<IllegalArgumentException>("2 bytes should be invalid") {
            ByteArrayConverter.convertBytesToIntArray(invalidBytes2)
        }
        assertFailsWith<IllegalArgumentException>("3 bytes should be invalid") {
            ByteArrayConverter.convertBytesToIntArray(invalidBytes3)
        }
        assertFailsWith<IllegalArgumentException>("7 bytes should be invalid") {
            ByteArrayConverter.convertBytesToIntArray(invalidBytes7)
        }
    }

    @Test
    fun testEmptyArrays() {
        // Test behavior with empty byte arrays
        val emptyBytes = ByteArray(0)
        
        val emptyFloats = ByteArrayConverter.convertBytesToFloatArray(emptyBytes)
        assertEquals(0, emptyFloats.size, "Empty byte array should produce empty float array")
        
        val emptyInts = ByteArrayConverter.convertBytesToIntArray(emptyBytes)
        assertEquals(0, emptyInts.size, "Empty byte array should produce empty int array")
        
        val emptyBytesResult = ByteArrayConverter.convertBytesToByteArray(emptyBytes)
        assertEquals(0, emptyBytesResult.size, "Empty byte array should remain empty")
    }

    @Test
    fun testSpecialFloatValues() {
        // Test conversion of special float values (NaN, infinity)
        val specialFloats = floatArrayOf(
            Float.NaN,
            Float.POSITIVE_INFINITY,
            Float.NEGATIVE_INFINITY,
            Float.MIN_VALUE,
            Float.MAX_VALUE
        )
        
        val bytes = ByteArray(20) // 5 floats * 4 bytes each
        for (i in specialFloats.indices) {
            val bits = specialFloats[i].toBits()
            bytes[i * 4 + 0] = (bits and 0xFF).toByte()
            bytes[i * 4 + 1] = ((bits shr 8) and 0xFF).toByte()
            bytes[i * 4 + 2] = ((bits shr 16) and 0xFF).toByte()
            bytes[i * 4 + 3] = ((bits shr 24) and 0xFF).toByte()
        }
        
        val converted = ByteArrayConverter.convertBytesToFloatArray(bytes)
        assertEquals(5, converted.size, "Should have 5 special floats")
        
        assertTrue(converted[0].isNaN(), "First value should be NaN")
        assertTrue(converted[1].isInfinite() && converted[1] > 0, "Second value should be positive infinity")
        assertTrue(converted[2].isInfinite() && converted[2] < 0, "Third value should be negative infinity")
        assertEquals(Float.MIN_VALUE, converted[3], "Fourth value should be MIN_VALUE")
        assertEquals(Float.MAX_VALUE, converted[4], "Fifth value should be MAX_VALUE")
    }

    @Test
    fun testExtremeIntValues() {
        // Test conversion of extreme int values
        val extremeInts = intArrayOf(
            Int.MIN_VALUE,
            Int.MAX_VALUE,
            0,
            -1,
            1
        )
        
        val bytes = ByteArray(20) // 5 ints * 4 bytes each
        for (i in extremeInts.indices) {
            bytes[i * 4 + 0] = (extremeInts[i] and 0xFF).toByte()
            bytes[i * 4 + 1] = ((extremeInts[i] shr 8) and 0xFF).toByte()
            bytes[i * 4 + 2] = ((extremeInts[i] shr 16) and 0xFF).toByte()
            bytes[i * 4 + 3] = ((extremeInts[i] shr 24) and 0xFF).toByte()
        }
        
        val converted = ByteArrayConverter.convertBytesToIntArray(bytes)
        assertEquals(5, converted.size, "Should have 5 extreme ints")
        
        for (i in extremeInts.indices) {
            assertEquals(extremeInts[i], converted[i], "Extreme int $i should match")
        }
    }

    @Test
    fun testLargeArrays() {
        // Test with larger arrays to ensure no performance issues
        val largeSize = 1000
        val largeFloatBytes = ByteArray(largeSize * 4)
        val largeIntBytes = ByteArray(largeSize * 4)
        
        // Fill with some pattern
        for (i in 0 until largeSize) {
            val floatBits = (i.toFloat()).toBits()
            largeFloatBytes[i * 4 + 0] = (floatBits and 0xFF).toByte()
            largeFloatBytes[i * 4 + 1] = ((floatBits shr 8) and 0xFF).toByte()
            largeFloatBytes[i * 4 + 2] = ((floatBits shr 16) and 0xFF).toByte()
            largeFloatBytes[i * 4 + 3] = ((floatBits shr 24) and 0xFF).toByte()
            
            largeIntBytes[i * 4 + 0] = (i and 0xFF).toByte()
            largeIntBytes[i * 4 + 1] = ((i shr 8) and 0xFF).toByte()
            largeIntBytes[i * 4 + 2] = ((i shr 16) and 0xFF).toByte()
            largeIntBytes[i * 4 + 3] = ((i shr 24) and 0xFF).toByte()
        }
        
        val convertedFloats = ByteArrayConverter.convertBytesToFloatArray(largeFloatBytes)
        val convertedInts = ByteArrayConverter.convertBytesToIntArray(largeIntBytes)
        
        assertEquals(largeSize, convertedFloats.size, "Should convert all floats")
        assertEquals(largeSize, convertedInts.size, "Should convert all ints")
        
        // Spot check a few values
        assertEquals(0.0f, convertedFloats[0], 0.001f, "First float should be 0.0")
        assertEquals(999.0f, convertedFloats[999], 0.001f, "Last float should be 999.0")
        assertEquals(0, convertedInts[0], "First int should be 0")
        assertEquals(999, convertedInts[999], "Last int should be 999")
    }
}