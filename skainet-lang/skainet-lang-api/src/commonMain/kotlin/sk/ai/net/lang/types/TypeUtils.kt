package sk.ai.net.lang.types

/**
 * Utility functions and extensions for the DType type system.
 * 
 * This object provides common operations, type hierarchy queries, and debugging utilities
 * for working with the tensor type system. It includes factory methods, type-safe builders,
 * and convenience functions for type compatibility and promotion logic.
 */
public object TypeUtils {
    
    /**
     * Finds the common promoted type for a collection of data types.
     * 
     * This function determines the most appropriate type that can represent
     * all input types without loss of precision. It applies promotion rules
     * transitively across all types in the collection.
     * 
     * @param types Collection of data types to find common promotion for
     * @return The common promoted type, or null if types are incompatible
     * @throws IllegalArgumentException if the collection is empty
     */
    public fun findCommonType(types: Collection<DType>): DType? {
        if (types.isEmpty()) {
            throw IllegalArgumentException("Cannot find common type for empty collection")
        }
        
        if (types.size == 1) {
            return types.first()
        }
        
        return types.reduce { acc, type ->
            if (acc.isCompatible(type)) {
                acc.promoteTo(type)
            } else {
                return null // Incompatible types found
            }
        }
    }
    
    /**
     * Finds the common promoted type for multiple data types (varargs version).
     * 
     * @param types Variable number of data types to find common promotion for
     * @return The common promoted type, or null if types are incompatible
     * @throws IllegalArgumentException if no types are provided
     */
    public fun findCommonType(vararg types: DType): DType? {
        return findCommonType(types.toList())
    }
    
    /**
     * Checks if all provided types are mutually compatible.
     * 
     * @param types Collection of data types to check
     * @return true if all types can be used together in operations, false otherwise
     */
    public fun areAllCompatible(types: Collection<DType>): Boolean {
        if (types.size <= 1) return true
        
        val typesList = types.toList()
        for (i in typesList.indices) {
            for (j in i + 1 until typesList.size) {
                if (!typesList[i].isCompatible(typesList[j])) {
                    return false
                }
            }
        }
        return true
    }
    
    /**
     * Checks if all provided types are mutually compatible (varargs version).
     * 
     * @param types Variable number of data types to check
     * @return true if all types can be used together in operations, false otherwise
     */
    public fun areAllCompatible(vararg types: DType): Boolean {
        return areAllCompatible(types.toList())
    }
    
    /**
     * Creates a type-safe builder for determining operation result types.
     * 
     * @param leftType The left operand type
     * @param rightType The right operand type
     * @return A TypePromotionBuilder for fluent API usage
     */
    public fun promote(leftType: DType, rightType: DType): TypePromotionBuilder {
        return TypePromotionBuilder(leftType, rightType)
    }
    
    /**
     * Builder class for type promotion operations with fluent API.
     */
    public class TypePromotionBuilder(
        private val leftType: DType,
        private val rightType: DType
    ) {
        /**
         * Executes the promotion and returns the result type.
         * 
         * @return The promoted type
         * @throws IllegalArgumentException if types are incompatible
         */
        public fun getResultType(): DType {
            if (!leftType.isCompatible(rightType)) {
                throw IllegalArgumentException(
                    "Types ${leftType.name} and ${rightType.name} are not compatible"
                )
            }
            return leftType.promoteTo(rightType)
        }
        
        /**
         * Executes the promotion and returns the result type, or null if incompatible.
         * 
         * @return The promoted type, or null if types are incompatible
         */
        public fun getResultTypeOrNull(): DType? {
            return if (leftType.isCompatible(rightType)) {
                leftType.promoteTo(rightType)
            } else {
                null
            }
        }
        
        /**
         * Checks if the promotion is possible without executing it.
         * 
         * @return true if promotion is possible, false otherwise
         */
        public fun isCompatible(): Boolean {
            return leftType.isCompatible(rightType)
        }
    }
    
    /**
     * Creates a detailed string representation of a data type with debugging information.
     * 
     * @param dtype The data type to describe
     * @return Detailed string with type information and compatibility rules
     */
    public fun describe(dtype: DType): String {
        val allTypes = DType.getAllTypes().values
        val compatibleTypes = allTypes.filter { dtype.isCompatible(it) }
        
        return buildString {
            appendLine("Type: ${dtype.name}")
            appendLine("Class: ${dtype::class.simpleName}")
            appendLine("Compatible with: ${compatibleTypes.joinToString { it.name }}")
            appendLine("Promotion rules:")
            compatibleTypes.forEach { other ->
                val promoted = dtype.promoteTo(other)
                appendLine("  ${dtype.name} + ${other.name} → ${promoted.name}")
            }
        }
    }
    
    /**
     * Prints debugging information for all registered data types.
     */
    public fun debugAllTypes() {
        val allTypes = DType.getAllTypes()
        println("=== Registered Data Types ===")
        allTypes.values.forEach { dtype ->
            println(describe(dtype))
            println()
        }
    }
    
    /**
     * Factory method to get a data type instance by name with type safety.
     * 
     * @param name The name of the data type
     * @return The data type instance
     * @throws IllegalArgumentException if the type is not found
     */
    public fun getTypeByName(name: String): DType {
        return DType.findByName(name) 
            ?: throw IllegalArgumentException("Unknown data type: $name")
    }
    
    /**
     * Checks if a given string represents a valid data type name.
     * 
     * @param name The name to check
     * @return true if the name corresponds to a registered data type
     */
    public fun isValidTypeName(name: String): Boolean {
        return DType.findByName(name) != null
    }
}