package sk.ainet.lang.tensor.dsl

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType
import kotlin.random.Random

/**
 * MLP-specific DSL extensions for creating neural network tensors
 */

/**
 * Creates weight tensors for MLP layers with common initialization strategies
 */
public class MLPWeights<T : DType, V>(private val builder: TensorBuilder<T, V>) {
    
    /**
     * Creates a weight matrix for a linear layer with Xavier/Glorot initialization
     * @param inputSize number of input features
     * @param outputSize number of output features
     */
    public fun xavier(inputSize: Int, outputSize: Int, random: Random = Random.Default): TensorInitializer<T, V> {
        val limit = kotlin.math.sqrt(6.0f / (inputSize + outputSize))
        return builder.shape(outputSize, inputSize)
            .uniform(-limit, limit, random)
    }
    
    /**
     * Creates a weight matrix for a linear layer with He initialization (good for ReLU)
     * @param inputSize number of input features
     * @param outputSize number of output features
     */
    public fun he(inputSize: Int, outputSize: Int, random: Random = Random.Default): TensorInitializer<T, V> {
        val std = kotlin.math.sqrt(2.0f / inputSize)
        return builder.shape(outputSize, inputSize)
            .randn(mean = 0.0f, std = std, random = random)
    }
    
    /**
     * Creates a weight matrix initialized with normal distribution
     * @param inputSize number of input features
     * @param outputSize number of output features
     */
    public fun normal(
        inputSize: Int, 
        outputSize: Int, 
        mean: Float = 0.0f, 
        std: Float = 0.01f, 
        random: Random = Random.Default
    ): TensorInitializer<T, V> {
        return builder.shape(outputSize, inputSize)
            .randn(mean, std, random)
    }
    
    /**
     * Creates a weight matrix initialized with zeros
     * @param inputSize number of input features
     * @param outputSize number of output features
     */
    public fun zeros(inputSize: Int, outputSize: Int): TensorInitializer<T, V> {
        return builder.shape(outputSize, inputSize).zeros()
    }
    
    /**
     * Creates a weight matrix initialized with ones
     * @param inputSize number of input features
     * @param outputSize number of output features
     */
    public fun ones(inputSize: Int, outputSize: Int): TensorInitializer<T, V> {
        return builder.shape(outputSize, inputSize).ones()
    }
}

/**
 * Creates bias tensors for MLP layers
 */
public class MLPBias<T : DType, V>(private val builder: TensorBuilder<T, V>) {
    
    /**
     * Creates a bias vector initialized with zeros (most common)
     * @param size number of output features
     */
    public fun zeros(size: Int): TensorInitializer<T, V> {
        return builder.shape(size).zeros()
    }
    
    /**
     * Creates a bias vector initialized with ones
     * @param size number of output features
     */
    public fun ones(size: Int): TensorInitializer<T, V> {
        return builder.shape(size).ones()
    }
    
    /**
     * Creates a bias vector initialized with a constant value
     * @param size number of output features
     * @param value constant value for initialization
     */
    public fun constant(size: Int, value: Number): TensorInitializer<T, V> {
        return builder.shape(size).fill(value)
    }
    
    /**
     * Creates a bias vector initialized with normal distribution
     * @param size number of output features
     */
    public fun normal(
        size: Int, 
        mean: Float = 0.0f, 
        std: Float = 0.01f, 
        random: Random = Random.Default
    ): TensorInitializer<T, V> {
        return builder.shape(size).randn(mean, std, random)
    }
}

/**
 * Extension function to access MLP weight creation
 */
public fun <T : DType, V> TensorBuilder<T, V>.weights(): MLPWeights<T, V> = MLPWeights(this)

/**
 * Extension function to access MLP bias creation
 */
public fun <T : DType, V> TensorBuilder<T, V>.bias(): MLPBias<T, V> = MLPBias(this)