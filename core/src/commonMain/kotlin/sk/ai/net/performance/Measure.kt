package sk.ai.net.performance

// Common source set
expect fun currentMillis(): Long

fun measureBlock(procedure: () -> Unit): Long {
    val start = currentMillis()
    procedure()
    val end = currentMillis()
    return end - start
}