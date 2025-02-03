package sk.ai.net.performance


actual fun currentMillis(): Long {
    throw NotImplementedError("Not implemented on wasm")
}