package sk.ai.net.nn.reflection

import sk.ai.net.nn.Module

// Extension function to generate a custom string representation for a Module
fun Module.toCustomString(indent: String = ""): String {
    // Build the string for the current module
    val builder = StringBuilder()
    builder.append("$indent$name")

    // If there are child modules, recursively add their representations with increased indent
    if (modules.isNotEmpty()) {
        modules.forEach { child ->
            builder.append("\n")
            builder.append(child.toCustomString("$indent    "))
        }
    }
    return builder.toString()
}

// Extension function for the root that prints the module tree hierarchy.
fun Module.toVisualString(): String {
    val builder = StringBuilder()
    // Print the root node (without any branch symbols)
    builder.append(name).append("\n")
    // For each child, call the helper function with an empty initial prefix.
    modules.forEachIndexed { index, module ->
        val isLast = index == modules.lastIndex
        builder.append(module.toVisualStringHelper("", isLast))
    }
    return builder.toString()
}

// Private helper extension function that handles the branch prefixes.
private fun Module.toVisualStringHelper(prefix: String, isLast: Boolean): String {
    val builder = StringBuilder()
    // Append the current prefix and the branch symbols:
    // "└── " if this node is the last child, otherwise "├── "
    builder.append(prefix)
    builder.append(if (isLast) "└── " else "├── ")
    builder.append(name)
    builder.append("\n")

    // Update the prefix for children:
    // If this node is the last, add spaces; otherwise add a vertical bar and spaces.
    val newPrefix = prefix + if (isLast) "    " else "│   "

    // Recursively process all child modules.
    modules.forEachIndexed { index, module ->
        val childIsLast = index == modules.lastIndex
        builder.append(module.toVisualStringHelper(newPrefix, childIsLast))
    }
    return builder.toString()
}
