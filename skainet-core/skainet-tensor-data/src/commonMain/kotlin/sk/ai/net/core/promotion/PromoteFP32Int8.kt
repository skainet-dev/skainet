package sk.ai.net.core.promotion

import sk.ainet.core.tensor.DType
import sk.ainet.core.tensor.FP32
import sk.ainet.core.tensor.Int8

/**
 * Concrete promotion strategy for FP32 and Int8 data types.
 *
 * This implementation handles promotion rules when FP32 is the first operand
 * and Int8 is the second operand in tensor operations. The promotion follows
 * the principle of precision preservation, where Int8 values are promoted
 * to FP32 to maintain floating-point precision.
 *
 * ## Supported Promotion Rules
 *
 * This strategy supports the following specific promotion cases:
 *
 * - **FP32 + FP32 → FP32** (same type, no promotion needed)
 * - **FP32 + Int8 → FP32** (Int8 promotes to FP32 for precision preservation)
 *
 * ## Thread Safety
 *
 * This implementation is thread-safe as it maintains no mutable state and
 * uses only immutable singleton objects for type comparisons and results.
 *
 * ## Edge Cases Handled
 *
 * 1. **Null Input Validation**: Throws IllegalArgumentException for null inputs
 * 2. **Unsupported Types**: Returns false for isPromotable() and throws for promote()
 * 3. **Same Type Operations**: Optimized handling for FP32 + FP32 operations
 *
 * @see DTypePromotion
 * @see PromoteInt8FP32
 * @see FP32
 * @see Int8
 *
 * @since 1.0.0
 */
public object PromoteFP32Int8 : DTypePromotion {

    /**
     * Determines the result type when promoting FP32 with another data type.
     *
     * This method implements the specific promotion logic for operations where
     * FP32 is involved as the first operand. The strategy prioritizes precision
     * preservation by promoting integer types to floating-point representation.
     *
     * ## Promotion Logic
     *
     * - If both types are FP32, returns FP32 (no promotion needed)
     * - If the second type is Int8, returns FP32 (promotes Int8 to FP32)
     * - All other combinations are considered incompatible
     *
     * ## Performance Considerations
     *
     * This method uses object identity checks (===) for type comparison,
     * which is more efficient than instanceof checks for singleton objects.
     *
     * @param a The first data type (expected to be FP32 for optimal behavior)
     * @param b The second data type to promote with
     * @return FP32 as the promoted type for all supported combinations
     * @throws IllegalArgumentException if the types cannot be promoted together
     * @throws IllegalArgumentException if either parameter is null
     *
     * @see isPromotable
     */
    override fun promote(a: DType, b: DType): DType {

        // Check if types are promotable before proceeding
        if (!isPromotable(a, b)) {
            throw IllegalArgumentException(
                "Types ${a.name} and ${b.name} are not compatible for promotion. " +
                        "Supported combinations: FP32+FP32, FP32+Int8"
            )
        }

        // All promotable combinations result in FP32 for this strategy
        return FP32
    }

    /**
     * Checks if FP32 can be promoted with another data type.
     *
     * This method determines compatibility for promotion operations involving
     * FP32 as one of the operands. The compatibility check is optimized for
     * the specific type combinations supported by this strategy.
     *
     * ## Compatibility Rules
     *
     * Returns true for the following combinations:
     * - FP32 + FP32 (same type operations)
     * - FP32 + Int8 (mixed precision operations)
     * - Int8 + FP32 (symmetric mixed precision operations)
     *
     * Returns false for:
     * - Null inputs
     * - Unsupported type combinations
     * - Future type extensions not yet implemented
     *
     * ## Implementation Notes
     *
     * This method is commutative by design, meaning isPromotable(FP32, Int8)
     * returns the same result as isPromotable(Int8, FP32). This ensures
     * consistent behavior regardless of operand order.
     *
     * @param a The first data type to check
     * @param b The second data type to check
     * @return true if the types can be promoted together, false otherwise
     *
     * @see promote
     */
    override fun isPromotable(a: DType, b: DType): Boolean {

        // Check for supported type combinations
        // This strategy supports: FP32+FP32, FP32+Int8, Int8+FP32
        return when {
            // Same type operations
            a === FP32 && b === FP32 -> true
            a === Int8 && b === Int8 -> false  // Not handled by this strategy

            // Mixed type operations (commutative)
            (a === FP32 && b === Int8) -> true
            (a === Int8 && b === FP32) -> true

            // Unsupported combinations
            else -> false
        }
    }

    /**
     * Returns a string representation of this promotion strategy.
     *
     * @return A descriptive name for this promotion strategy
     */
    override fun toString(): String = "PromoteFP32Int8[FP32+{FP32,Int8} → FP32]"
}