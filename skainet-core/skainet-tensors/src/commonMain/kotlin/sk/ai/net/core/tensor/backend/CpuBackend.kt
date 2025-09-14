package sk.ai.net.core.tensor.backend

import sk.ai.net.core.tensor.*

/**
 * A CPU-based tensor for FP32/Float values.
 *
 * This tensor stores data on the CPU using simple FloatArray with NCHW row-major layout. 
 * It supports 1-4 dimensional tensors and delegates all operations to CpuBackend.
 */
public class CpuTensorFP32(
    override val shape: Shape,
    internal val data: FloatArray
) : Tensor<FP32, Float> {
    
    init {
        require(data.size == shape.volume) {
            "Data size ${data.size} doesn't match shape volume ${shape.volume}"
        }
        require(shape.rank in 1..4) {
            "Only 1-4 dimensional tensors are supported, got ${shape.rank}"
        }
    }
    
    override fun get(vararg indices: Int): Float {
        val index = shape.index(indices)
        return data[index]
    }
    
    // Delegate all operations to the CpuBackend
    private val backend = CpuBackend()
    
    override fun matmul(a: Tensor<FP32, Float>, b: Tensor<FP32, Float>): Tensor<FP32, Float> = backend.matmul(a, b)
    override fun scale(a: Tensor<FP32, Float>, scalar: Double): Tensor<FP32, Float> = backend.scale(a, scalar)
    override fun dot(a: Tensor<FP32, Float>, b: Tensor<FP32, Float>): Double = backend.dot(a, b)
    
    // Tensor-Tensor operations
    override fun Tensor<FP32, Float>.plus(other: Tensor<FP32, Float>): Tensor<FP32, Float> = with(backend) { this@plus.plus(other) }
    override fun Tensor<FP32, Float>.minus(other: Tensor<FP32, Float>): Tensor<FP32, Float> = with(backend) { this@minus.minus(other) }
    override fun Tensor<FP32, Float>.times(other: Tensor<FP32, Float>): Tensor<FP32, Float> = with(backend) { this@times.times(other) }
    override fun Tensor<FP32, Float>.div(other: Tensor<FP32, Float>): Tensor<FP32, Float> = with(backend) { this@div.div(other) }
    
    // Tensor-Scalar operations - delegate to backend
    override fun Tensor<FP32, Float>.plus(scalar: Int): Tensor<FP32, Float> = with(backend) { this@plus.plus(scalar) }
    override fun Tensor<FP32, Float>.minus(scalar: Int): Tensor<FP32, Float> = with(backend) { this@minus.minus(scalar) }
    override fun Tensor<FP32, Float>.times(scalar: Int): Tensor<FP32, Float> = with(backend) { this@times.times(scalar) }
    override fun Tensor<FP32, Float>.div(scalar: Int): Tensor<FP32, Float> = with(backend) { this@div.div(scalar) }
    
    override fun Tensor<FP32, Float>.plus(scalar: Float): Tensor<FP32, Float> = with(backend) { this@plus.plus(scalar) }
    override fun Tensor<FP32, Float>.minus(scalar: Float): Tensor<FP32, Float> = with(backend) { this@minus.minus(scalar) }
    override fun Tensor<FP32, Float>.times(scalar: Float): Tensor<FP32, Float> = with(backend) { this@times.times(scalar) }
    override fun Tensor<FP32, Float>.div(scalar: Float): Tensor<FP32, Float> = with(backend) { this@div.div(scalar) }
    
    override fun Tensor<FP32, Float>.plus(scalar: Double): Tensor<FP32, Float> = with(backend) { this@plus.plus(scalar) }
    override fun Tensor<FP32, Float>.minus(scalar: Double): Tensor<FP32, Float> = with(backend) { this@minus.minus(scalar) }
    override fun Tensor<FP32, Float>.times(scalar: Double): Tensor<FP32, Float> = with(backend) { this@times.times(scalar) }
    override fun Tensor<FP32, Float>.div(scalar: Double): Tensor<FP32, Float> = with(backend) { this@div.div(scalar) }
    
    // Scalar-Tensor operations - delegate to backend
    override fun Double.plus(t: Tensor<FP32, Float>): Tensor<FP32, Float> = with(backend) { this@plus.plus(t) }
    override fun Double.minus(t: Tensor<FP32, Float>): Tensor<FP32, Float> = with(backend) { this@minus.minus(t) }
    override fun Double.times(t: Tensor<FP32, Float>): Tensor<FP32, Float> = with(backend) { this@times.times(t) }
    override fun Double.div(t: Tensor<FP32, Float>): Tensor<FP32, Float> = with(backend) { this@div.div(t) }
    
    public companion object {
        /**
         * Creates a tensor from an array with the given shape.
         */
        public fun fromArray(shape: Shape, data: FloatArray): CpuTensorFP32 {
            return CpuTensorFP32(shape, data)
        }
        
        /**
         * Creates a tensor filled with zeros.
         */
        public fun zeros(shape: Shape): CpuTensorFP32 {
            return CpuTensorFP32(shape, FloatArray(shape.volume))
        }
        
        /**
         * Creates a tensor filled with ones.
         */
        public fun ones(shape: Shape): CpuTensorFP32 {
            return CpuTensorFP32(shape, FloatArray(shape.volume) { 1.0f })
        }
        
        /**
         * Creates a tensor filled with a specific value.
         */
        public fun full(shape: Shape, value: Float): CpuTensorFP32 {
            return CpuTensorFP32(shape, FloatArray(shape.volume) { value })
        }
    }
}

/**
 * A CPU-based implementation of the ComputeBackend interface for FP32/Float tensors.
 */
public class CpuBackend : ComputeBackend<FP32, Float> {
    override val name: String = "CPU"
    
    // Basic operations - implement the actual computation logic
    override fun matmul(a: Tensor<FP32, Float>, b: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(a is CpuTensorFP32 && b is CpuTensorFP32) { "Both tensors must be CpuTensorFP32" }
        require(a.shape.rank == 2 && b.shape.rank == 2) { "Matrix multiplication requires 2D tensors" }
        require(a.shape[1] == b.shape[0]) { "Matrix dimensions don't match for multiplication" }
        
        val rows = a.shape[0]
        val cols = b.shape[1]
        val inner = a.shape[1]
        val result = FloatArray(rows * cols)
        
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                var sum = 0f
                for (k in 0 until inner) {
                    sum += a.data[i * inner + k] * b.data[k * cols + j]
                }
                result[i * cols + j] = sum
            }
        }
        
        return CpuTensorFP32(Shape(rows, cols), result)
    }
    
    override fun scale(a: Tensor<FP32, Float>, scalar: Double): Tensor<FP32, Float> {
        require(a is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = a.data.map { it * scalar.toFloat() }.toFloatArray()
        return CpuTensorFP32(a.shape, result)
    }
    
    override fun dot(a: Tensor<FP32, Float>, b: Tensor<FP32, Float>): Double {
        require(a is CpuTensorFP32 && b is CpuTensorFP32) { "Both tensors must be CpuTensorFP32" }
        require(a.shape == b.shape) { "Tensors must have same shape for dot product" }
        
        var sum = 0.0
        for (i in a.data.indices) {
            sum += a.data[i] * b.data[i]
        }
        return sum
    }
    
    // Tensor-Tensor operations - implement actual computation logic
    override fun Tensor<FP32, Float>.plus(other: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(this is CpuTensorFP32 && other is CpuTensorFP32) { "Both tensors must be CpuTensorFP32" }
        require(this.shape == other.shape) { "Tensors must have same shape for addition" }
        
        val result = FloatArray(this.data.size)
        for (i in this.data.indices) {
            result[i] = this.data[i] + other.data[i]
        }
        return CpuTensorFP32(this.shape, result)
    }
    
    override fun Tensor<FP32, Float>.minus(other: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(this is CpuTensorFP32 && other is CpuTensorFP32) { "Both tensors must be CpuTensorFP32" }
        require(this.shape == other.shape) { "Tensors must have same shape for subtraction" }
        
        val result = FloatArray(this.data.size)
        for (i in this.data.indices) {
            result[i] = this.data[i] - other.data[i]
        }
        return CpuTensorFP32(this.shape, result)
    }
    
    override fun Tensor<FP32, Float>.times(other: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(this is CpuTensorFP32 && other is CpuTensorFP32) { "Both tensors must be CpuTensorFP32" }
        require(this.shape == other.shape) { "Tensors must have same shape for element-wise multiplication" }
        
        val result = FloatArray(this.data.size)
        for (i in this.data.indices) {
            result[i] = this.data[i] * other.data[i]
        }
        return CpuTensorFP32(this.shape, result)
    }
    
    override fun Tensor<FP32, Float>.div(other: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(this is CpuTensorFP32 && other is CpuTensorFP32) { "Both tensors must be CpuTensorFP32" }
        require(this.shape == other.shape) { "Tensors must have same shape for element-wise division" }
        
        val result = FloatArray(this.data.size)
        for (i in this.data.indices) {
            result[i] = this.data[i] / other.data[i]
        }
        return CpuTensorFP32(this.shape, result)
    }
    
    // Tensor-Scalar operations
    override fun Tensor<FP32, Float>.plus(scalar: Int): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { it + scalar }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }
    
    override fun Tensor<FP32, Float>.minus(scalar: Int): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { it - scalar }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }
    
    override fun Tensor<FP32, Float>.times(scalar: Int): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { it * scalar }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }
    
    override fun Tensor<FP32, Float>.div(scalar: Int): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { it / scalar }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }
    
    override fun Tensor<FP32, Float>.plus(scalar: Float): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { it + scalar }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }
    
    override fun Tensor<FP32, Float>.minus(scalar: Float): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { it - scalar }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }
    
    override fun Tensor<FP32, Float>.times(scalar: Float): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { it * scalar }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }
    
    override fun Tensor<FP32, Float>.div(scalar: Float): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { it / scalar }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }
    
    override fun Tensor<FP32, Float>.plus(scalar: Double): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { it + scalar.toFloat() }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }
    
    override fun Tensor<FP32, Float>.minus(scalar: Double): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { it - scalar.toFloat() }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }
    
    override fun Tensor<FP32, Float>.times(scalar: Double): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { it * scalar.toFloat() }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }
    
    override fun Tensor<FP32, Float>.div(scalar: Double): Tensor<FP32, Float> {
        require(this is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = this.data.map { it / scalar.toFloat() }.toFloatArray()
        return CpuTensorFP32(this.shape, result)
    }
    
    // Scalar-Tensor operations
    override fun Double.plus(t: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(t is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = t.data.map { this.toFloat() + it }.toFloatArray()
        return CpuTensorFP32(t.shape, result)
    }
    
    override fun Double.minus(t: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(t is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = t.data.map { this.toFloat() - it }.toFloatArray()
        return CpuTensorFP32(t.shape, result)
    }
    
    override fun Double.times(t: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(t is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = t.data.map { this.toFloat() * it }.toFloatArray()
        return CpuTensorFP32(t.shape, result)
    }
    
    override fun Double.div(t: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(t is CpuTensorFP32) { "Tensor must be CpuTensorFP32" }
        val result = t.data.map { this.toFloat() / it }.toFloatArray()
        return CpuTensorFP32(t.shape, result)
    }
}