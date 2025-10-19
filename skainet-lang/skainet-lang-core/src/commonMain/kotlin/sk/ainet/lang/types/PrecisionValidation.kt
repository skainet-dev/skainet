package sk.ainet.lang.types

/**
 * Compile-time validation utilities for precision compatibility.
 * 
 * This file provides utility functions for validating precision compatibility
 * at compile time, helping developers catch precision-related errors early.
 */

/**
 * Validates that two DTypes are compatible for operations.
 * 
 * @param source The source data type
 * @param target The target data type
 * @throws IllegalArgumentException if the types are not compatible
 */
public fun validateCompatibility(source: DType, target: DType) {
    if (!source.isCompatible(target)) {
        throw IllegalArgumentException(
            "DType ${source.name} is not compatible with ${target.name}. " +
            "Compatible pairs include: FP32↔FP16, Int32↔Int8, etc."
        )
    }
}

/**
 * Validates that a conversion path exists between two DTypes.
 * 
 * @param from The source data type
 * @param to The target data type
 * @throws IllegalArgumentException if conversion is not supported
 */
public fun validateConversion(from: DType, to: DType) {
    if (!from.isConvertibleTo(to)) {
        throw IllegalArgumentException(
            "Conversion from ${from.name} to ${to.name} is not supported. " +
            "Supported conversions include precision promotions and safe downcasts."
        )
    }
}

/**
 * Validates that a promotion between two DTypes will produce the expected result.
 * 
 * @param type1 The first data type
 * @param type2 The second data type
 * @param expectedResult The expected promoted type
 * @throws IllegalArgumentException if promotion doesn't match expected result
 */
public fun validatePromotion(type1: DType, type2: DType, expectedResult: DType) {
    val actualResult = type1.promoteTo(type2)
    if (actualResult != expectedResult) {
        throw IllegalArgumentException(
            "Promotion of ${type1.name} and ${type2.name} produced ${actualResult.name}, " +
            "but expected ${expectedResult.name}"
        )
    }
}

/**
 * Checks if a precision chain is valid (e.g., for multi-stage networks).
 * 
 * @param precisionChain List of DTypes representing the precision flow
 * @return true if the chain is valid, false otherwise
 */
public fun isValidPrecisionChain(precisionChain: List<DType>): Boolean {
    if (precisionChain.isEmpty()) return true
    
    return precisionChain.zipWithNext().all { (current, next) ->
        current.isCompatible(next)
    }
}

/**
 * Validates an entire precision chain, throwing an exception if invalid.
 * 
 * @param precisionChain List of DTypes representing the precision flow
 * @throws IllegalArgumentException if the chain contains incompatible transitions
 */
public fun validatePrecisionChain(precisionChain: List<DType>) {
    if (!isValidPrecisionChain(precisionChain)) {
        val invalidTransitions = precisionChain.zipWithNext()
            .filterNot { (current, next) -> current.isCompatible(next) }
            .map { (current, next) -> "${current.name} -> ${next.name}" }
        
        throw IllegalArgumentException(
            "Precision chain contains invalid transitions: ${invalidTransitions.joinToString(", ")}"
        )
    }
}