package sk.ainet.core.tensor.factory

import sk.ainet.core.tensor.DefaultTensorFactories

/**
 * Initializes DefaultTensorFactories with mock factory implementations.
 * This provides a clean way to set up the default factories for testing and development.
 */
public object DefaultTensorFactoryInitializer {
    
    /**
     * Flag to track if default factories have been initialized
     */
    private var initialized = false
    
    /**
     * Initializes DefaultTensorFactories with mock implementations.
     * This method is thread-safe and can be called multiple times safely.
     */
    public fun initializeDefaultFactories() {
        if (initialized) {
            return
        }
        
        // Set up the mock factories in DefaultTensorFactories
        DefaultTensorFactories.setFP32Factory(MockTensorFactoryFP32)
        DefaultTensorFactories.setInt8Factory(MockTensorFactoryInt8)
        DefaultTensorFactories.setInt32Factory(MockTensorFactoryInt32)
        
        initialized = true
    }
    
    /**
     * Checks if the default factories have been initialized.
     * @return true if factories are initialized, false otherwise
     */
    public fun isInitialized(): Boolean = initialized
    
    /**
     * Resets the initialization state (primarily for testing).
     */
    internal fun reset() {
        initialized = false
    }
}