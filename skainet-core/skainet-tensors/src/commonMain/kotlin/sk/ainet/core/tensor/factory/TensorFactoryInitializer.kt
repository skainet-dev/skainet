package sk.ainet.core.tensor.factory

import sk.ainet.core.tensor.FP32
import sk.ainet.core.tensor.Int8
import sk.ainet.core.tensor.Int32

/**
 * Initializes all tensor factories and registers them in the TensorFactoryRegistry.
 * This object should be initialized during application startup to make all factories available.
 */
public object TensorFactoryInitializer {
    
    /**
     * Flag to track if factories have been initialized
     */
    private var initialized = false
    
    /**
     * Initializes and registers all tensor factories.
     * This method is thread-safe and can be called multiple times safely.
     */
    public fun initializeFactories() {
        if (initialized) {
            return
        }
        
        // Register FP32 factory
        TensorFactoryRegistry.registerFactory<FP32>(FP32TensorFactory)
        
        // Register Int8 factory  
        TensorFactoryRegistry.registerFactory<Int8>(Int8TensorFactory)
        
        // Register Int32 factory
        TensorFactoryRegistry.registerFactory<Int32>(Int32TensorFactory)
        
        initialized = true
    }
    
    /**
     * Checks if the factories have been initialized.
     * @return true if factories are initialized, false otherwise
     */
    public fun isInitialized(): Boolean = initialized
    
    /**
     * Resets the initialization state and clears all registered factories.
     * This method is primarily for testing purposes.
     */
    internal fun reset() {
        TensorFactoryRegistry.clearFactories()
        initialized = false
    }
    
    /**
     * Gets the count of registered factories.
     * @return Number of registered factories
     */
    public fun getRegisteredFactoryCount(): Int {
        return TensorFactoryRegistry.getRegisteredDTypes().size
    }
    
    /**
     * Prints information about registered factories (for debugging).
     */
    public fun printRegisteredFactories() {
        val registeredTypes = TensorFactoryRegistry.getRegisteredDTypes()
        println("Registered tensor factories:")
        registeredTypes.forEach { dtypeClass ->
            println("  - ${dtypeClass.simpleName}")
        }
        println("Total: ${registeredTypes.size} factories")
    }
}