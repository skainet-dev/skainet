# skainet-lang Module

## Overview
The `skainet-lang` module is the core language module for the SKaiNET neural network framework. It serves as the parent module containing various submodules that provide neural network computation capabilities, memory management, and performance optimization components.

## Architecture
This module acts as an umbrella for multiple specialized submodules:
- **skainet-lang-api** - Neural network scripting API with data types and tensor operations
- **skainet-lang-memory** - Memory management and data structures for neural computations
- **skainet-performance** - Performance measurement and optimization utilities
- **skainet-tensor-ops** - Tensor operation implementations
- **skainet-tensors** - Core tensor functionality
- **skainet-tensors-api** - Tensor API definitions

## Dependencies
- Kotlin Multiplatform (Common, JVM, Native, Android, iOS, WASM)
- Gradle build system

## Purpose
The skainet-lang module provides:
- Core neural network language constructs
- Memory-efficient data structures
- Cross-platform tensor operations
- Performance optimization tools
- Unified API for neural network computations

## Usage
This module is typically not used directly but through its submodules. Each submodule provides specific functionality:

```kotlin
// Example usage through submodules
import sk.ai.net.lang.types.* // from skainet-lang-api
import sk.ai.net.lang.memory.* // from skainet-lang-memory
```

## Submodules
- `skainet-lang-api` - Core API and data types
- `skainet-lang-memory` - Memory management layer
- `skainet-performance` - Performance utilities
- `skainet-tensor-ops` - Tensor operations
- `skainet-tensors` - Tensor implementations
- `skainet-tensors-api` - Tensor API

## Testing
Tests are distributed across individual submodules. Run all tests with:
```bash
./gradlew :skainet-lang:test
```

## Contributing
When contributing to skainet-lang:
1. Identify the appropriate submodule for your changes
2. Follow the coding standards of the specific submodule
3. Ensure cross-platform compatibility
4. Add appropriate tests for new functionality