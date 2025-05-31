package sk.ai.net.io.data.mnist

/**
 * WASM JS implementation of the MNIST loader factory.
 */
actual object MNISTLoaderFactory {
    /**
     * Creates a new MNIST loader with the default configuration.
     *
     * @return A new MNIST loader.
     */
    actual fun create(): MNISTLoader {
        return MNISTLoaderWasmJs.create()
    }

    /**
     * Creates a new MNIST loader with a custom cache directory.
     *
     * @param cacheDir The directory to use for caching.
     * @return A new MNIST loader.
     */
    actual fun create(cacheDir: String): MNISTLoader {
        return MNISTLoaderWasmJs.create(cacheDir)
    }

    /**
     * Creates a new MNIST loader with a custom configuration.
     *
     * @param config The configuration to use.
     * @return A new MNIST loader.
     */
    actual fun create(config: MNISTLoaderConfig): MNISTLoader {
        return MNISTLoaderWasmJs.create(config)
    }
}