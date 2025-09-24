package sk.ainet.core.tensor

/**
 * Provides default TensorFactory instances for supported DType and value type combinations.
 * This enables automatic factory resolution in DSL contexts without requiring explicit factory parameters.
 * 
 * The implementation uses type-based resolution to return the appropriate backend factory.
 */
public object DefaultTensorFactories {
    
    /**
     * Internal storage for factory instances to avoid recreating them.
     */
    private var fp32Factory: TensorFactory<FP32, Float>? = null
    private var int8Factory: TensorFactory<Int8, Byte>? = null  
    private var int32Factory: TensorFactory<Int32, Int>? = null
    
    /**
     * Sets the FP32 factory implementation. This is called by the implementation module.
     */
    public fun setFP32Factory(factory: TensorFactory<FP32, Float>) {
        fp32Factory = factory
    }
    
    /**
     * Sets the Int8 factory implementation. This is called by the implementation module.
     */
    public fun setInt8Factory(factory: TensorFactory<Int8, Byte>) {
        int8Factory = factory
    }
    
    /**
     * Sets the Int32 factory implementation. This is called by the implementation module.
     */
    public fun setInt32Factory(factory: TensorFactory<Int32, Int>) {
        int32Factory = factory
    }
    
    /**
     * Gets the default TensorFactory for FP32/Float combinations.
     */
    public fun getFP32Factory(): TensorFactory<FP32, Float> {
        return fp32Factory ?: throw IllegalStateException(
            "FP32 factory not initialized. Make sure the tensor implementation module is included."
        )
    }
    
    /**
     * Gets the default TensorFactory for Int8/Byte combinations.
     */
    public fun getInt8Factory(): TensorFactory<Int8, Byte> {
        return int8Factory ?: throw IllegalStateException(
            "Int8 factory not initialized. Make sure the tensor implementation module is included."
        )
    }
    
    /**
     * Gets the default TensorFactory for Int32/Int combinations.
     */
    public fun getInt32Factory(): TensorFactory<Int32, Int> {
        return int32Factory ?: throw IllegalStateException(
            "Int32 factory not initialized. Make sure the tensor implementation module is included."
        )
    }
}