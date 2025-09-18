package sk.ainet.core.tensor

/**
 * Generic extension function to print scalar tensors (1D with single element).
 */
public fun <T : DType, V> Tensor<T, V>.printScalar(): String {
    require(shape.rank == 1 && shape[0] == 1) { 
        "Tensor must be a scalar (1D with single element), got rank ${shape.rank} with shape ${shape.dimensions.contentToString()}" 
    }
    return this[0].toString()
}

/**
 * Generic extension function to print vector tensors (1D).
 * Returns a string representation in the format [a, b, c, ...].
 */
public fun <T : DType, V> Tensor<T, V>.printVector(): String {
    require(shape.rank == 1) { "Tensor must be a vector (1D), got rank ${shape.rank}" }
    val elements = (0 until shape[0]).map { this[it].toString() }
    return "[${elements.joinToString(", ")}]"
}

/**
 * Generic extension function to print matrix tensors (2D).
 * Returns a string representation with each row on a separate line.
 */
public fun <T : DType, V> Tensor<T, V>.printMatrix(): String {
    require(shape.rank == 2) { "Tensor must be a matrix (2D), got rank ${shape.rank}" }
    val rows = shape[0]
    val cols = shape[1]
    val result = StringBuilder()

    result.append("[\n")
    for (i in 0 until rows) {
        result.append("  [")
        for (j in 0 until cols) {
            result.append(this[i, j].toString())
            if (j < cols - 1) result.append(", ")
        }
        result.append("]")
        if (i < rows - 1) result.append(",")
        result.append("\n")
    }
    result.append("]")

    return result.toString()
}

/**
 * Generic extension function to print 3D tensors.
 * Useful for batch processing, RGB images, etc.
 */
public fun <T : DType, V> Tensor<T, V>.print3D(): String {
    require(shape.rank == 3) { "Tensor must be 3D, got rank ${shape.rank}" }
    val dim0 = shape[0]
    val dim1 = shape[1] 
    val dim2 = shape[2]
    val result = StringBuilder()

    result.append("[\n")
    for (i in 0 until dim0) {
        result.append("  [\n")
        for (j in 0 until dim1) {
            result.append("    [")
            for (k in 0 until dim2) {
                result.append(this[i, j, k].toString())
                if (k < dim2 - 1) result.append(", ")
            }
            result.append("]")
            if (j < dim1 - 1) result.append(",")
            result.append("\n")
        }
        result.append("  ]")
        if (i < dim0 - 1) result.append(",")
        result.append("\n")
    }
    result.append("]")

    return result.toString()
}

/**
 * Generic extension function to iterate over all tensor elements.
 * Useful for custom printing formats or processing all elements.
 */
public fun <T : DType, V> Tensor<T, V>.forEachIndexed(action: (indices: IntArray, value: V) -> Unit) {
    fun recursiveIterate(currentIndices: IntArray, dimensionIndex: Int) {
        if (dimensionIndex == shape.rank) {
            // Base case: we've built a complete index
            val value = when (currentIndices.size) {
                1 -> this@forEachIndexed[currentIndices[0]]
                2 -> this@forEachIndexed[currentIndices[0], currentIndices[1]]
                3 -> this@forEachIndexed[currentIndices[0], currentIndices[1], currentIndices[2]]
                4 -> this@forEachIndexed[currentIndices[0], currentIndices[1], currentIndices[2], currentIndices[3]]
                else -> throw UnsupportedOperationException("Tensors with more than 4 dimensions are not supported for iteration")
            }
            action(currentIndices, value)
            return
        }
        
        // Recursive case: iterate through current dimension
        for (i in 0 until shape[dimensionIndex]) {
            val newIndices = currentIndices + i
            recursiveIterate(newIndices, dimensionIndex + 1)
        }
    }
    
    recursiveIterate(intArrayOf(), 0)
}

/**
 * General extension function to print tensors of any dimension.
 * Automatically selects the appropriate printing method based on tensor rank.
 */
public fun <T : DType, V> Tensor<T, V>.print(): String {
    return when (shape.rank) {
        1 if shape[0] == 1 -> printScalar()
        1 -> printVector()
        2 -> printMatrix()
        3 -> print3D()
        else -> {
            // For higher dimensions, provide a summary
            val elementCount = minOf(10, shape.volume) // Show first 10 elements
            val elements = mutableListOf<V>()
            var count = 0
            
            forEachIndexed { _, value ->
                if (count < elementCount) {
                    elements.add(value)
                    count++
                }
            }
            
            val preview = elements.joinToString(", ") { it.toString() }
            val more = if (shape.volume > elementCount) ", ..." else ""
            "Tensor(shape=${shape.dimensions.contentToString()}, rank=${shape.rank}) [$preview$more]"
        }
    }
}