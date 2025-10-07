package sk.ai.net.lang.types

import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.assertFalse
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertFailsWith

class DTypeTest {

    @Test
    fun testCompatibilityChecks() {
        // Test positive compatibility cases
        assertTrue(Ternary.isCompatible(Int8), "Ternary should be compatible with Int8")
        assertTrue(Int4.isCompatible(Int8), "Int4 should be compatible with Int8")
        assertTrue(Int8.isCompatible(FP32), "Int8 should be compatible with FP32")
        assertTrue(FP16.isCompatible(FP32), "FP16 should be compatible with FP32")
        assertTrue(Int32.isCompatible(Int8), "Int32 should be compatible with Int8")

        // Test same type compatibility
        assertTrue(Int8.isCompatible(Int8), "Int8 should be compatible with itself")
        assertTrue(FP32.isCompatible(FP32), "FP32 should be compatible with itself")

        // Test bidirectional compatibility
        assertTrue(Int8.isCompatible(Int32), "Int8 should be compatible with Int32")
        assertTrue(Int32.isCompatible(Int8), "Int32 should be compatible with Int8")
    }

    @Test
    fun testPromotionRules() {
        // Test basic promotion cases
        assertEquals(Int8, Ternary.promoteTo(Int8), "Ternary + Int8 should promote to Int8")
        assertEquals(Int8, Int4.promoteTo(Int8), "Int4 + Int8 should promote to Int8")
        assertEquals(FP32, Int8.promoteTo(FP32), "Int8 + FP32 should promote to FP32")
        assertEquals(FP32, FP16.promoteTo(FP32), "FP16 + FP32 should promote to FP32")
        assertEquals(FP32, Int32.promoteTo(FP32), "Int32 + FP32 should promote to FP32")
        assertEquals(FP32, FP32.promoteTo(Int8), "FP32 + Int8 should promote to FP32")

        // Test same type promotion
        assertEquals(Int8, Int8.promoteTo(Int8), "Int8 + Int8 should remain Int8")
        assertEquals(FP32, FP32.promoteTo(FP32), "FP32 + FP32 should remain FP32")

        // Test hierarchy promotion
        assertEquals(Int32, Int8.promoteTo(Int32), "Int8 + Int32 should promote to Int32")
        assertEquals(FP16, Int8.promoteTo(FP16), "Int8 + FP16 should promote to FP16")
    }

    @Test
    fun testRegistryFunctionality() {
        val allTypes = DType.getAllTypes()

        // Test that registry contains expected types
        assertTrue(allTypes.containsKey("Ternary"), "Registry should contain Ternary")
        assertTrue(allTypes.containsKey("Int4"), "Registry should contain Int4")
        assertTrue(allTypes.containsKey("Int8"), "Registry should contain Int8")
        assertTrue(allTypes.containsKey("Int32"), "Registry should contain Int32")
        assertTrue(allTypes.containsKey("Float16"), "Registry should contain Float16")
        assertTrue(allTypes.containsKey("Float32"), "Registry should contain Float32")

        // Test findByName functionality
        val fp32ByName = DType.findByName("Float32")
        assertNotNull(fp32ByName, "Should find FP32 by name")
        assertEquals("Float32", fp32ByName.name, "Found type should have correct name")
        assertEquals(FP32, fp32ByName, "Found type should be the same instance as FP32")

        val unknownType = DType.findByName("Unknown")
        assertNull(unknownType, "Should return null for unknown type names")
    }

    @Test
    fun testTypeUtilsFunctionality() {
        // Test findCommonType
        val commonType = TypeUtils.findCommonType(Int8, FP16, FP32)
        assertNotNull(commonType, "Should find common type for Int8, FP16, FP32")
        assertEquals(FP32, commonType, "Common type should be FP32")

        // Test areAllCompatible
        val areCompatible = TypeUtils.areAllCompatible(Ternary, Int4, Int8, FP32)
        assertFalse(areCompatible, "All types (Ternary, Int4, Int8, FP32) are not compatible")

        // Test type description (should not throw)
        val description = TypeUtils.describe(Int8)
        assertTrue(description.contains("Int8"), "Description should contain type name")
        assertTrue(description.contains("Compatible with:"), "Description should contain compatibility info")
    }

    @Test
    fun testTypeUtilsEdgeCases() {
        // Test empty collection
        assertFailsWith<IllegalArgumentException> {
            TypeUtils.findCommonType(emptyList())
        }

        // Test single type
        val singleType = TypeUtils.findCommonType(listOf(Int8))
        assertEquals(Int8, singleType, "Single type should return itself")

        // Test single type with varargs
        val singleTypeVarargs = TypeUtils.findCommonType(FP32)
        assertEquals(FP32, singleTypeVarargs, "Single type varargs should return itself")

        // Test areAllCompatible with single type
        assertTrue(TypeUtils.areAllCompatible(Int8), "Single type should be compatible with itself")

        // Test areAllCompatible with empty collection
        assertTrue(TypeUtils.areAllCompatible(emptyList()), "Empty collection should be compatible")
    }

    @Test
    fun testTypeUtilsPromotionBuilder() {
        val builder = TypeUtils.promote(Int8, FP32)

        assertTrue(builder.isCompatible(), "Int8 and FP32 should be compatible via builder")
        assertEquals(FP32, builder.getResultType(), "Promotion should result in FP32")
        assertEquals(FP32, builder.getResultTypeOrNull(), "Promotion should result in FP32 (null-safe)")
    }

    @Test
    fun testTypeUtilsGetTypeByName() {
        val int8ByName = TypeUtils.getTypeByName("Int8")
        assertEquals(Int8, int8ByName, "Should get Int8 by name")

        assertFailsWith<IllegalArgumentException> {
            TypeUtils.getTypeByName("NonExistent")
        }
    }

    @Test
    fun testTypeUtilsValidTypeName() {
        assertTrue(TypeUtils.isValidTypeName("Int8"), "Int8 should be a valid type name")
        assertTrue(TypeUtils.isValidTypeName("Float32"), "Float32 should be a valid type name")
        assertFalse(TypeUtils.isValidTypeName("NonExistent"), "NonExistent should not be a valid type name")
    }

    @Test
    fun testTypeSizeProperties() {
        assertEquals(2, Ternary.sizeInBits, "Ternary should be 2 bits")
        assertEquals(4, Int4.sizeInBits, "Int4 should be 4 bits")
        assertEquals(8, Int8.sizeInBits, "Int8 should be 8 bits")
        assertEquals(32, Int32.sizeInBits, "Int32 should be 32 bits")
        assertEquals(16, FP16.sizeInBits, "FP16 should be 16 bits")
        assertEquals(32, FP32.sizeInBits, "FP32 should be 32 bits")
    }

    @Test
    fun testTypeNameProperties() {
        assertEquals("Ternary", Ternary.name, "Ternary should have correct name")
        assertEquals("Int4", Int4.name, "Int4 should have correct name")
        assertEquals("Int8", Int8.name, "Int8 should have correct name")
        assertEquals("Int32", Int32.name, "Int32 should have correct name")
        assertEquals("Float16", FP16.name, "FP16 should have correct name")
        assertEquals("Float32", FP32.name, "FP32 should have correct name")
    }
}