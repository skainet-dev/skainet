package sk.ai.net.core.promotion

import sk.ainet.core.tensor.DType


/**
 * Interface for defining type promotion strategies in the tensor system.
 * 
 * Type promotion is a fundamental aspect of tensor operations that determines how
 * different data types interact when used together in mathematical operations.
 * This interface provides a pluggable strategy system for implementing various
 * promotion rules while maintaining consistency and extensibility.
 * 
 * ## Promotion Rules Overview
 * 
 * The promotion system follows these general principles:
 * 
 * 1. **Commutative Promotion**: promote(A, B) must equal promote(B, A)
 * 2. **Precision Preservation**: Promotion should preserve or increase precision
 * 3. **Consistency**: If isPromotable(A, B) returns true, promote(A, B) must succeed
 * 4. **Transitivity**: If A promotes to C and B promotes to C, then promote(A, B) should equal C
 * 
 * ## Current Type Hierarchy
 * 
 * The current implementation supports the following promotion rules:
 * 
 * - **FP32 + FP32 → FP32** (same type, no promotion needed)
 * - **Int8 + Int8 → Int8** (same type, no promotion needed)  
 * - **FP32 + Int8 → FP32** (Int8 promotes to FP32 for precision)
 * - **Int8 + FP32 → FP32** (symmetric to above)
 * 
 * ## Implementation Guidelines
 * 
 * When implementing this interface:
 * 
 * 1. Ensure promote() is commutative: promote(a, b) == promote(b, a)
 * 2. Validate compatibility before promotion
 * 3. Handle edge cases gracefully with appropriate exceptions
 * 4. Document specific promotion rules in the implementation
 * 5. Consider future extensibility for new data types
 * 
 * ## Thread Safety
 * 
 * Implementations should be thread-safe as promotion strategies may be used
 * concurrently across multiple tensor operations in multiplatform environments.
 * 
 * @see DType
 * @see PromotionRegistry
 * 
 * @since 1.0.0
 */
public interface DTypePromotion {
    
    /**
     * Determines the result type when promoting two data types.
     * 
     * This method defines how mixed-type operations should be handled by determining
     * the appropriate result type that can represent values from both input types
     * without loss of precision where possible.
     * 
     * The promotion must be commutative, meaning promote(a, b) must return the same
     * result as promote(b, a). This ensures consistent behavior regardless of operand
     * order in mathematical operations.
     * 
     * ## Examples
     * 
     * ```kotlin
     * val promotion = SomePromotionImplementation()
     * val result1 = promotion.promote(FP32, Int8) // Returns FP32
     * val result2 = promotion.promote(Int8, FP32) // Returns FP32 (same as result1)
     * ```
     * 
     * @param a The first data type to promote
     * @param b The second data type to promote
     * @return The promoted data type that can represent both input types
     * @throws IllegalArgumentException if the types cannot be promoted together
     * @throws IllegalStateException if the promotion strategy is in an invalid state
     * 
     * @see isPromotable
     */
    public fun promote(a: DType, b: DType): DType
    
    /**
     * Checks if two data types can be promoted together for operations.
     * 
     * This method determines whether the two given data types are compatible
     * for promotion and can be used together in mathematical operations.
     * If this method returns true, then promote(a, b) must succeed without
     * throwing an exception.
     * 
     * The compatibility check must also be commutative: isPromotable(a, b)
     * must return the same result as isPromotable(b, a).
     * 
     * ## Implementation Notes
     * 
     * - Always call this method before promote() to avoid exceptions
     * - Return false for null or unsupported type combinations
     * - Consider future extensibility when implementing compatibility rules
     * 
     * ## Examples
     * 
     * ```kotlin
     * val promotion = SomePromotionImplementation()
     * 
     * if (promotion.isPromotable(FP32, Int8)) {
     *     val resultType = promotion.promote(FP32, Int8) // Safe to call
     * }
     * ```
     * 
     * @param a The first data type to check
     * @param b The second data type to check
     * @return true if the types can be promoted together, false otherwise
     * 
     * @see promote
     */
    public fun isPromotable(a: DType, b: DType): Boolean
}

/**
 * Sealed class hierarchy for representing promotion results.
 * 
 * This provides a more type-safe alternative to throwing exceptions
 * and allows for better error handling and pattern matching in promotion
 * operations. Implementations may choose to use this instead of exceptions
 * for more functional programming approaches.
 * 
 * @param T The promoted data type
 */
public sealed class PromotionResult<out T : DType> {
    
    /**
     * Successful promotion result containing the promoted type.
     * 
     * @property promotedType The resulting data type from successful promotion
     */
    public data class Success<T : DType>(val promotedType: T) : PromotionResult<T>()
    
    /**
     * Failed promotion result with error information.
     * 
     * @property reason Human-readable description of why promotion failed
     * @property sourceType The first type that failed to promote
     * @property targetType The second type that failed to promote
     */
    public data class Failure(
        val reason: String,
        val sourceType: DType,
        val targetType: DType
    ) : PromotionResult<Nothing>()
}