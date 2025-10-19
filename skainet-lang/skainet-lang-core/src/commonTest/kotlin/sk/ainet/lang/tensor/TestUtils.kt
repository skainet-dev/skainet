package sk.ainet.lang.tensor

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.types.DType
import kotlin.random.Random
import kotlin.reflect.KClass

// Mock TensorData for testing
class MockTensorData<T : DType, V>(
    override val shape: Shape,
    private val data: FloatArray
) : TensorData<T, V> {
    @Suppress("UNCHECKED_CAST")
    override fun get(vararg indices: Int): V = data[shape.index(indices)] as V
    @Suppress("UNCHECKED_CAST")
    override fun set(vararg indices: Int, value: V) { 
        data[shape.index(indices)] = (value as Number).toFloat()
    }
}

// Simple mock factory for testing
val testFactory = object : TensorDataFactory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> zeros(shape: Shape, dtype: KClass<T>): TensorData<T, V> = 
        MockTensorData<T, V>(shape, FloatArray(shape.volume) { 0.0f })
        
    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> ones(shape: Shape, dtype: KClass<T>): TensorData<T, V> = 
        MockTensorData<T, V>(shape, FloatArray(shape.volume) { 1.0f })
        
    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> full(shape: Shape, dtype: KClass<T>, value: Number): TensorData<T, V> = 
        MockTensorData<T, V>(shape, FloatArray(shape.volume) { value.toFloat() })
        
    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> randn(
        shape: Shape,
        dtype: KClass<T>,
        mean: Float,
        std: Float,
        random: Random
    ): TensorData<T, V> = 
        MockTensorData<T, V>(shape, FloatArray(shape.volume) { 
            // Simple normal distribution approximation
            val u1 = random.nextFloat()
            val u2 = random.nextFloat()
            val z = kotlin.math.sqrt(-2.0 * kotlin.math.ln(u1.toDouble())) * kotlin.math.cos(2.0 * kotlin.math.PI * u2.toDouble())
            (z.toFloat() * std + mean)
        })
        
    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> uniform(
        shape: Shape,
        dtype: KClass<T>,
        min: Float,
        max: Float,
        random: Random
    ): TensorData<T, V> = 
        MockTensorData<T, V>(shape, FloatArray(shape.volume) { random.nextFloat() * (max - min) + min })
        
    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> init(
        shape: Shape,
        dtype: KClass<T>,
        generator: (indices: IntArray) -> V
    ): TensorData<T, V> = 
        MockTensorData<T, V>(shape, FloatArray(shape.volume) { index ->
            val indices = computeIndices(index, shape)
            (generator(indices) as Number).toFloat()
        })
        
    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> randomInit(
        shape: Shape,
        dtype: KClass<T>,
        generator: (random: Random) -> V,
        random: Random
    ): TensorData<T, V> = 
        MockTensorData<T, V>(shape, FloatArray(shape.volume) { (generator(random) as Number).toFloat() })
        
    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> fromFloatArray(
        shape: Shape,
        dtype: KClass<T>,
        data: FloatArray
    ): TensorData<T, V> = 
        MockTensorData<T, V>(shape, data.copyOf())
        
    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> fromIntArray(
        shape: Shape,
        dtype: KClass<T>,
        data: IntArray
    ): TensorData<T, V> = 
        MockTensorData<T, V>(shape, data.map { it.toFloat() }.toFloatArray())
        
    private fun computeIndices(flatIndex: Int, shape: Shape): IntArray {
        val indices = IntArray(shape.rank)
        var remaining = flatIndex
        for (i in shape.rank - 1 downTo 0) {
            indices[i] = remaining % shape[i]
            remaining /= shape[i]
        }
        return indices
    }
}