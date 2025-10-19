# skainet-lang-api Module

## Overview
The `skainet-lang-api` module provides the core neural network scripting API for the SKaiNET framework. It defines fundamental data types, type utilities, and tensor data structures used throughout the neural network computation system.

## Architecture
This module serves as the foundational API layer that:
- Defines core data types for neural network computations
- Provides type conversion utilities
- Establishes tensor data interfaces
- Supports cross-platform neural network operations

## Dependencies
- Kotlin Multiplatform (Common, JVM, Native targets)
- Parent module: `skainet-lang`

## Core Components

### Data Types
The module provides various numerical data types optimized for neural network computations:
- **FP32** - 32-bit floating point operations
- **FP16** - 16-bit floating point for memory efficiency
- **Int32** - 32-bit integer operations
- **Int8** - 8-bit integer for quantized networks
- **Int4** - 4-bit integer for extreme quantization
- **Ternary** - Ternary values for specialized networks

### Type Utilities
- **DType** - Core data type abstraction
- **DTypeExtensions** - Extension functions for data type operations
- **TypeUtils** - Utility functions for type conversions and operations

### Tensor Data
- **TensorData** - The fundamental data structure for tensor operations in the SKaiNET framework. This core abstraction provides unified interface for multi-dimensional data arrays used in neural network computations, serving as the foundation for weight storage, activation containers, gradient computations, and input/output representations.
- **ItemsAccessor** - Base interface providing indexed access to elements with multi-dimensional indexing support

## Public API
Key classes and interfaces exposed by this module:
```kotlin
// Core data types
DType, FP32, FP16, Int32, Int8, Int4, Ternary

// Tensor data structures
TensorData

// Utility functions
TypeUtils, DTypeExtensions
```

## Usage Examples
```kotlin
// Working with data types
val fp32Type = FP32()
val int8Type = Int8()

// Type conversions
val convertedValue = TypeUtils.convert(value, targetType)

// Tensor data operations
val tensorView = ViewTensorData(...)
```

## Platform Support
- **Common** - Core multiplatform implementation
- **JVM** - Java Virtual Machine specific optimizations
- **Native** - Native platform implementations

## Testing
Run tests with:
```bash
./gradlew :skainet-lang:skainet-lang-api:test
```

## Contributing
When contributing to skainet-lang-api:
1. Ensure new data types follow the established patterns
2. Maintain cross-platform compatibility
3. Add comprehensive tests for new functionality
4. Document public API changes
5. Consider performance implications of new features