package sk.ai.net.core.promotion

import sk.ainet.core.tensor.DType
import sk.ainet.core.tensor.FP32
import sk.ainet.core.tensor.Int8


/**
 * Registry for type promotion strategies in the tensor system.
 *
 * This class provides a centralized registry for managing promotion strategies
 * and performing type promotion lookups in a thread-safe manner. The registry
 * supports both explicit strategy registration and automatic fallback to
 * default promotion rules.
 *
 * ## Registry Pattern
 *
 * The registry implements the strategy pattern with centralized lookup,
 * allowing for:
 *
 * 1. **Dynamic Strategy Registration**: Register custom promotion strategies
 * 2. **Type Pair Lookup**: Efficient lookup by source and target type pairs
 * 3. **Default Fallbacks**: Built-in promotion strategies for common types
 * 4. **Thread Safety**: Concurrent access from multiple tensor operations
 * 5. **Lazy Initialization**: Strategies are loaded only when needed
 *
 * ## Default Promotion Strategies
 *
 * The registry comes pre-configured with the following promotion strategies:
 *
 * - **FP32 + Int8**: Handled by PromoteFP32Int8
 * - **Int8 + FP32**: Handled by PromoteInt8FP32
 * - **Same Type Operations**: Handled by respective strategies
 *
 * ## Thread Safety
 *
 * This implementation provides basic thread safety for multiplatform environments.
 * While not using platform-specific concurrent collections, the simple operations
 * and immutable strategy objects provide reasonable thread safety for most use cases.
 *
 * @see DTypePromotion
 * @see PromoteFP32Int8
 * @see PromoteInt8FP32
 *
 * @since 1.0.0
 */
public class PromotionRegistry {

    /**
     * Storage for custom promotion strategies.
     * Uses a type pair key for efficient lookup.
     */
    private val customStrategies: MutableMap<TypePair, DTypePromotion> = mutableMapOf()

    /**
     * Lazy-initialized default promotion strategies.
     * These are loaded only when first accessed to optimize startup time.
     */
    private val defaultStrategies: Map<TypePair, DTypePromotion> by lazy {
        buildMap {
            // FP32 strategies
            put(TypePair(FP32::class, FP32::class), PromoteFP32Int8)
            put(TypePair(FP32::class, Int8::class), PromoteFP32Int8)

            // Int8 strategies  
            put(TypePair(Int8::class, Int8::class), PromoteInt8FP32)
            put(TypePair(Int8::class, FP32::class), PromoteInt8FP32)
        }
    }

    /**
     * Registers a promotion strategy for a specific type pair.
     *
     * This method allows registration of custom promotion strategies that will
     * take precedence over default strategies. Custom strategies override default
     * strategies for the same type pair.
     *
     * ## Registration Rules
     *
     * - Custom strategies override default strategies for the same type pair
     * - Registration is permanent for the lifetime of the registry instance
     * - Both forward and reverse type pairs should be registered for symmetry
     * - Null strategies or types will result in IllegalArgumentException
     *
     * ## Example Usage
     *
     * ```kotlin
     * val registry = PromotionRegistry()
     * registry.register(FP32::class, Int8::class, CustomPromotionStrategy())
     * registry.register(Int8::class, FP32::class, CustomPromotionStrategy()) // Symmetry
     * ```
     *
     * @param sourceType The source data type class
     * @param targetType The target data type class
     * @param strategy The promotion strategy to use for this type pair
     * @throws IllegalArgumentException if any parameter is null
     *
     * @see getPromotion
     */
    public fun register(
        sourceType: kotlin.reflect.KClass<out DType>,
        targetType: kotlin.reflect.KClass<out DType>,
        strategy: DTypePromotion
    ) {
        val typePair = TypePair(sourceType, targetType)
        customStrategies[typePair] = strategy
    }

    /**
     * Retrieves the promotion strategy for a specific type pair.
     *
     * This method performs efficient lookup of promotion strategies, first
     * checking registered custom strategies, then falling back to default
     * strategies if no custom strategy is found.
     *
     * ## Lookup Strategy
     *
     * 1. Check registered custom strategies first
     * 2. Fall back to default strategies if no custom strategy found
     * 3. Return null if no strategy is available for the type pair
     *
     * @param sourceType The source data type class
     * @param targetType The target data type class
     * @return The promotion strategy for the type pair, or null if none exists
     * @throws IllegalArgumentException if any parameter is null
     *
     * @see register
     */
    public fun getPromotion(
        sourceType: kotlin.reflect.KClass<out DType>,
        targetType: kotlin.reflect.KClass<out DType>
    ): DTypePromotion? {
        val typePair = TypePair(sourceType, targetType)

        // Check registered custom strategies first
        customStrategies[typePair]?.let { return it }

        // Fall back to default strategies
        return defaultStrategies[typePair]
    }

    /**
     * Promotes two data types using the registered promotion strategies.
     *
     * This is a convenience method that combines strategy lookup and promotion
     * in a single operation. It automatically handles strategy selection and
     * delegates to the appropriate promotion implementation.
     *
     * ## Promotion Process
     *
     * 1. Look up the appropriate promotion strategy for the type pair
     * 2. Delegate to the strategy's promote() method
     * 3. Return the promoted type or throw an exception if promotion fails
     *
     * @param a The first data type to promote
     * @param b The second data type to promote
     * @return The promoted data type
     * @throws IllegalArgumentException if no promotion strategy is available
     * @throws IllegalArgumentException if the promotion strategy rejects the types
     *
     * @see isPromotable
     */
    public fun promote(a: DType, b: DType): DType {
        val strategy = getPromotion(a::class, b::class)
            ?: throw IllegalArgumentException(
                "No promotion strategy available for types: ${a.name} and ${b.name}"
            )

        return strategy.promote(a, b)
    }

    /**
     * Checks if two data types can be promoted using registered strategies.
     *
     * This method provides a safe way to check type compatibility before
     * attempting promotion, helping to avoid exceptions in performance-critical
     * code paths.
     *
     * @param a The first data type to check
     * @param b The second data type to check
     * @return true if the types can be promoted, false otherwise
     *
     * @see promote
     */
    public fun isPromotable(a: DType, b: DType): Boolean {
        val strategy = getPromotion(a::class, b::class) ?: return false
        return strategy.isPromotable(a, b)
    }

    /**
     * Gets all registered promotion strategies.
     *
     * This method returns a read-only snapshot of all currently registered
     * promotion strategies, including both custom and default strategies.
     * The returned map is a copy and modifications will not affect the registry.
     *
     * @return A read-only map of all promotion strategies
     */
    public fun getAllStrategies(): Map<TypePair, DTypePromotion> {
        return buildMap {
            putAll(defaultStrategies)
            putAll(customStrategies) // Custom strategies override defaults
        }
    }

    /**
     * Clears all custom registered strategies, keeping only defaults.
     *
     * This method is useful for testing or resetting the registry to its
     * initial state. Default strategies cannot be removed and will remain
     * available after clearing.
     */
    public fun clearCustomStrategies() {
        customStrategies.clear()
    }

    public companion object {
        /**
         * Singleton instance of the promotion registry.
         *
         * This provides a convenient global access point for promotion operations
         * while still allowing custom registry instances when needed for isolation
         * or testing purposes.
         */
        public val default: PromotionRegistry by lazy { PromotionRegistry() }
    }
}

/**
 * Represents a pair of data types for use as a map key.
 *
 * This class provides efficient hashing and equality for type pairs,
 * enabling fast lookup of promotion strategies in the registry.
 *
 * @property sourceType The source data type class
 * @property targetType The target data type class
 */
public data class TypePair(
    val sourceType: kotlin.reflect.KClass<out DType>,
    val targetType: kotlin.reflect.KClass<out DType>
) {

    override fun hashCode(): Int {
        return sourceType.hashCode() * 31 + targetType.hashCode()
    }

    override fun equals(other: Any?): Boolean {
        return other is TypePair &&
                sourceType == other.sourceType &&
                targetType == other.targetType
    }

    override fun toString(): String {
        return "${sourceType.simpleName} -> ${targetType.simpleName}"
    }
}