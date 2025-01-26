package sk.ai.net

class Shape(vararg dimensions: Int) {
    val dimensions: IntArray = dimensions.copyOf()

    val volume: Int
        get() = dimensions.fold(1) { a, x -> a * x }

    val rank: Int
        get() = dimensions.size
}