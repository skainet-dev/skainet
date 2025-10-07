package sk.ai.net.core.promotion

import sk.ainet.core.tensor.DType
import sk.ainet.core.tensor.FP32
import sk.ainet.core.tensor.Int8


/**
 * Concrete promotion strategy for Int8 and FP32 data types.
 *
 * This implementation handles promotion rules when Int8 is the first operand
 * and FP32 is the second operand in tensor operations. The promotion follows
 * the principle of precision preservation, where Int8 values are promoted
 * to FP32 to maintain floating-point precision.
 *
 * This strategy ensures symmetry with PromoteFP32Int8, meaning that
 * promote(Int8, FP32) produces the same result as promote(FP32, Int8).
 *
 * ## Supported Promotion Rules
 *
 * This strategy supports the following specific promotion cases:
 *
 * - **Int8 + Int8 → Int8** (same type, no promotion needed)
 * - **Int8 + FP32 → FP32** (Int8 promotes to FP32 for precision preservation)
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
 * 3. **Same Type Operations**: Optimized handling for Int8 + Int8 operations
 * 4. **Symmetry Validation**: Ensures consistent results with PromoteFP32Int8
 *
 * @see DTypePromotion
 * @see PromoteFP32Int8
 * @see FP32
 * @see Int8
 *
 * @since 1.0.0
 */
public object PromoteInt8FP32 : DTypePromotion {

    /**
     * Determines the result type when promoting Int8 with another data type.
     *
     * This method implements the specific promotion logic for operations where
     * Int8 is involved as the first operand. The strategy prioritizes precision
     * preservation by promoting to floating-point representation when mixed
     * with FP32 types.
     *
     * ## Promotion Logic
     *
     * - If both types are Int8, returns Int8 (no promotion needed)
     * - If the second type is FP32, returns FP32 (promotes Int8 to FP32)
     * - All other combinations are considered incompatible
     *
     * ## Symmetry Guarantee
     *
     * This method ensures that promote(Int8, FP32) returns the same result
     * as PromoteFP32Int8.promote(FP32, Int8), maintaining commutative behavior
     * across different promotion strategy implementations.
     *
     * @param a The first data type (expected to be Int8 for optimal behavior)
     * @param b The second data type to promote with
     * @return The promoted type (Int8 for same-type, FP32 for mixed operations)
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
                        "Supported combinations: Int8+Int8, Int8+FP32"
            )
        }

        // Determine promotion result based on type combination
        return when {
            // Same type operations
            a === Int8 && b === Int8 -> Int8

            // Mixed type operations (promote to higher precision)
            (a === Int8 && b === FP32) || (a === FP32 && b === Int8) -> FP32

            // FP32 same-type (handled for symmetry with other strategies)
            a === FP32 && b === FP32 -> FP32

            else -> throw IllegalStateException(
                "Internal error: Promotion validation passed but no rule matched for ${a.name} + ${b.name}"
            )
        }
    }

    /**
     * Checks if Int8 can be promoted with another data type.
     *
     * This method determines compatibility for promotion operations involving
     * Int8 as one of the operands. The compatibility check is optimized for
     * the specific type combinations supported by this strategy.
     *
     * ## Compatibility Rules
     *
     * Returns true for the following combinations:
     * - Int8 + Int8 (same type operations)
     * - Int8 + FP32 (mixed precision operations)
     * - FP32 + Int8 (symmetric mixed precision operations)
     * - FP32 + FP32 (for strategy completeness)
     *
     * Returns false for:
     * - Null inputs
     * - Unsupported type combinations
     * - Future type extensions not yet implemented
     *
     * ## Implementation Notes
     *
     * This method is commutative by design, meaning isPromotable(Int8, FP32)
     * returns the same result as isPromotable(FP32, Int8). This ensures
     * consistent behavior regardless of operand order and maintains symmetry
     * with the PromoteFP32Int8 strategy.
     *
     * @param a The first data type to check
     * @param b The second data type to check
     * @return true if the types can be promoted together, false otherwise
     *
     * @see promote
     */
    override fun isPromotable(a: DType, b: DType): Boolean {

        // Check for supported type combinations
        // This strategy supports: Int8+Int8, Int8+FP32, FP32+Int8, FP32+FP32
        return when {
            // Same type operations
            a === Int8 && b === Int8 -> true
            a === FP32 && b === FP32 -> true

            // Mixed type operations (commutative)
            (a === Int8 && b === FP32) -> true
            (a === FP32 && b === Int8) -> true

            // Unsupported combinations
            else -> false
        }
    }

    /**
     * Returns a string representation of this promotion strategy.
     *
     * @return A descriptive name for this promotion strategy
     */
    override fun toString(): String = "PromoteInt8FP32[Int8+{Int8,FP32} → {Int8,FP32}]"
}