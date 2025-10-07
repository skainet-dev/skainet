package sk.ainet.core.tensor.factory

import sk.ainet.core.tensor.*
import sk.ainet.core.tensor.backend.*
import kotlin.random.Random

/**
 * Mock tensor factory implementations that create tensors with real data but mock operations.
 * These factories enable tensor creation from real data sources (files, arrays) while
 * keeping all tensor operations unimplemented for separation of concerns.
 */

/**
 * Mock factory for creating FP32 tensors with real data but mock operations.
 */
public object MockTensorFactoryFP32 : TensorFactory<FP32, Float> {
    
    override fun zeros(shape: Shape): Tensor<FP32, Float> {
        return MockTensorFP32.zeros(shape)
    }

    override fun ones(shape: Shape): Tensor<FP32, Float> {
        return MockTensorFP32.ones(shape)
    }

    override fun random(shape: Shape): Tensor<FP32, Float> {
        val data = FloatArray(shape.volume) { Random.nextFloat() }
        return MockTensorFP32.fromArray(shape, data)
    }

    override fun random(shape: Shape, seed: Long): Tensor<FP32, Float> {
        val random = Random(seed)
        val data = FloatArray(shape.volume) { random.nextFloat() }
        return MockTensorFP32.fromArray(shape, data)
    }

    override fun random(shape: Shape, random: Random): Tensor<FP32, Float> {
        val data = FloatArray(shape.volume) { random.nextFloat() }
        return MockTensorFP32.fromArray(shape, data)
    }

    override fun randomNormal(shape: Shape, mean: Double, std: Double): Tensor<FP32, Float> {
        // Use Box-Muller transform for normal distribution approximation
        val data = FloatArray(shape.volume) { 
            val u1 = Random.nextDouble()
            val u2 = Random.nextDouble()
            val normal = kotlin.math.sqrt(-2.0 * kotlin.math.ln(u1)) * kotlin.math.cos(2.0 * kotlin.math.PI * u2)
            (normal * std + mean).toFloat()
        }
        return MockTensorFP32.fromArray(shape, data)
    }

    override fun randomNormal(shape: Shape, mean: Double, std: Double, seed: Long): Tensor<FP32, Float> {
        val random = Random(seed)
        val data = FloatArray(shape.volume) { 
            val u1 = random.nextDouble()
            val u2 = random.nextDouble()
            val normal = kotlin.math.sqrt(-2.0 * kotlin.math.ln(u1)) * kotlin.math.cos(2.0 * kotlin.math.PI * u2)
            (normal * std + mean).toFloat()
        }
        return MockTensorFP32.fromArray(shape, data)
    }

    override fun randomNormal(shape: Shape, mean: Double, std: Double, random: Random): Tensor<FP32, Float> {
        val data = FloatArray(shape.volume) { 
            val u1 = random.nextDouble()
            val u2 = random.nextDouble()
            val normal = kotlin.math.sqrt(-2.0 * kotlin.math.ln(u1)) * kotlin.math.cos(2.0 * kotlin.math.PI * u2)
            (normal * std + mean).toFloat()
        }
        return MockTensorFP32.fromArray(shape, data)
    }

    override fun randomUniform(shape: Shape, min: Double, max: Double): Tensor<FP32, Float> {
        val data = FloatArray(shape.volume) { 
            (Random.nextDouble(min, max)).toFloat()
        }
        return MockTensorFP32.fromArray(shape, data)
    }

    override fun randomUniform(shape: Shape, min: Double, max: Double, seed: Long): Tensor<FP32, Float> {
        val random = Random(seed)
        val data = FloatArray(shape.volume) { 
            (random.nextDouble(min, max)).toFloat()
        }
        return MockTensorFP32.fromArray(shape, data)
    }

    override fun randomUniform(shape: Shape, min: Double, max: Double, random: Random): Tensor<FP32, Float> {
        val data = FloatArray(shape.volume) { 
            (random.nextDouble(min, max)).toFloat()
        }
        return MockTensorFP32.fromArray(shape, data)
    }

    override fun fromArray(shape: Shape, data: FloatArray): Tensor<FP32, Float> {
        return MockTensorFP32.fromArray(shape, data)
    }

    override fun fromArray(shape: Shape, data: IntArray): Tensor<FP32, Float> {
        val floatData = FloatArray(data.size) { data[it].toFloat() }
        return MockTensorFP32.fromArray(shape, floatData)
    }
}

/**
 * Mock factory for creating Int32 tensors with real data but mock operations.
 */
public object MockTensorFactoryInt32 : TensorFactory<Int32, Int> {
    
    override fun zeros(shape: Shape): Tensor<Int32, Int> {
        return MockTensorInt32.zeros(shape)
    }

    override fun ones(shape: Shape): Tensor<Int32, Int> {
        return MockTensorInt32.ones(shape)
    }

    override fun random(shape: Shape): Tensor<Int32, Int> {
        val data = IntArray(shape.volume) { Random.nextInt() }
        return MockTensorInt32.fromArray(shape, data)
    }

    override fun random(shape: Shape, seed: Long): Tensor<Int32, Int> {
        val random = Random(seed)
        val data = IntArray(shape.volume) { random.nextInt() }
        return MockTensorInt32.fromArray(shape, data)
    }

    override fun random(shape: Shape, random: Random): Tensor<Int32, Int> {
        val data = IntArray(shape.volume) { random.nextInt() }
        return MockTensorInt32.fromArray(shape, data)
    }

    override fun randomNormal(shape: Shape, mean: Double, std: Double): Tensor<Int32, Int> {
        val data = IntArray(shape.volume) { 
            val u1 = Random.nextDouble()
            val u2 = Random.nextDouble()
            val normal = kotlin.math.sqrt(-2.0 * kotlin.math.ln(u1)) * kotlin.math.cos(2.0 * kotlin.math.PI * u2)
            (normal * std + mean).toInt()
        }
        return MockTensorInt32.fromArray(shape, data)
    }

    override fun randomNormal(shape: Shape, mean: Double, std: Double, seed: Long): Tensor<Int32, Int> {
        val random = Random(seed)
        val data = IntArray(shape.volume) { 
            val u1 = random.nextDouble()
            val u2 = random.nextDouble()
            val normal = kotlin.math.sqrt(-2.0 * kotlin.math.ln(u1)) * kotlin.math.cos(2.0 * kotlin.math.PI * u2)
            (normal * std + mean).toInt()
        }
        return MockTensorInt32.fromArray(shape, data)
    }

    override fun randomNormal(shape: Shape, mean: Double, std: Double, random: Random): Tensor<Int32, Int> {
        val data = IntArray(shape.volume) { 
            val u1 = random.nextDouble()
            val u2 = random.nextDouble()
            val normal = kotlin.math.sqrt(-2.0 * kotlin.math.ln(u1)) * kotlin.math.cos(2.0 * kotlin.math.PI * u2)
            (normal * std + mean).toInt()
        }
        return MockTensorInt32.fromArray(shape, data)
    }

    override fun randomUniform(shape: Shape, min: Double, max: Double): Tensor<Int32, Int> {
        val data = IntArray(shape.volume) { 
            Random.nextDouble(min, max).toInt()
        }
        return MockTensorInt32.fromArray(shape, data)
    }

    override fun randomUniform(shape: Shape, min: Double, max: Double, seed: Long): Tensor<Int32, Int> {
        val random = Random(seed)
        val data = IntArray(shape.volume) { 
            random.nextDouble(min, max).toInt()
        }
        return MockTensorInt32.fromArray(shape, data)
    }

    override fun randomUniform(shape: Shape, min: Double, max: Double, random: Random): Tensor<Int32, Int> {
        val data = IntArray(shape.volume) { 
            random.nextDouble(min, max).toInt()
        }
        return MockTensorInt32.fromArray(shape, data)
    }

    override fun fromArray(shape: Shape, data: FloatArray): Tensor<Int32, Int> {
        val intData = IntArray(data.size) { data[it].toInt() }
        return MockTensorInt32.fromArray(shape, intData)
    }

    override fun fromArray(shape: Shape, data: IntArray): Tensor<Int32, Int> {
        return MockTensorInt32.fromArray(shape, data)
    }
}

/**
 * Mock factory for creating Int8 tensors with real data but mock operations.
 */
public object MockTensorFactoryInt8 : TensorFactory<Int8, Byte> {
    
    override fun zeros(shape: Shape): Tensor<Int8, Byte> {
        return MockTensorInt8.zeros(shape)
    }

    override fun ones(shape: Shape): Tensor<Int8, Byte> {
        val data = ByteArray(shape.volume) { 1 }
        return MockTensorInt8.fromArray(shape, data)
    }

    override fun random(shape: Shape): Tensor<Int8, Byte> {
        val data = ByteArray(shape.volume) { Random.nextInt(-128, 127).toByte() }
        return MockTensorInt8.fromArray(shape, data)
    }

    override fun random(shape: Shape, seed: Long): Tensor<Int8, Byte> {
        val random = Random(seed)
        val data = ByteArray(shape.volume) { random.nextInt(-128, 127).toByte() }
        return MockTensorInt8.fromArray(shape, data)
    }

    override fun random(shape: Shape, random: Random): Tensor<Int8, Byte> {
        val data = ByteArray(shape.volume) { random.nextInt(-128, 127).toByte() }
        return MockTensorInt8.fromArray(shape, data)
    }

    override fun randomNormal(shape: Shape, mean: Double, std: Double): Tensor<Int8, Byte> {
        val data = ByteArray(shape.volume) { 
            val u1 = Random.nextDouble()
            val u2 = Random.nextDouble()
            val normal = kotlin.math.sqrt(-2.0 * kotlin.math.ln(u1)) * kotlin.math.cos(2.0 * kotlin.math.PI * u2)
            (normal * std + mean).toInt().coerceIn(-128, 127).toByte()
        }
        return MockTensorInt8.fromArray(shape, data)
    }

    override fun randomNormal(shape: Shape, mean: Double, std: Double, seed: Long): Tensor<Int8, Byte> {
        val random = Random(seed)
        val data = ByteArray(shape.volume) { 
            val u1 = random.nextDouble()
            val u2 = random.nextDouble()
            val normal = kotlin.math.sqrt(-2.0 * kotlin.math.ln(u1)) * kotlin.math.cos(2.0 * kotlin.math.PI * u2)
            (normal * std + mean).toInt().coerceIn(-128, 127).toByte()
        }
        return MockTensorInt8.fromArray(shape, data)
    }

    override fun randomNormal(shape: Shape, mean: Double, std: Double, random: Random): Tensor<Int8, Byte> {
        val data = ByteArray(shape.volume) { 
            val u1 = random.nextDouble()
            val u2 = random.nextDouble()
            val normal = kotlin.math.sqrt(-2.0 * kotlin.math.ln(u1)) * kotlin.math.cos(2.0 * kotlin.math.PI * u2)
            (normal * std + mean).toInt().coerceIn(-128, 127).toByte()
        }
        return MockTensorInt8.fromArray(shape, data)
    }

    override fun randomUniform(shape: Shape, min: Double, max: Double): Tensor<Int8, Byte> {
        val data = ByteArray(shape.volume) { 
            Random.nextDouble(min, max).toInt().coerceIn(-128, 127).toByte()
        }
        return MockTensorInt8.fromArray(shape, data)
    }

    override fun randomUniform(shape: Shape, min: Double, max: Double, seed: Long): Tensor<Int8, Byte> {
        val random = Random(seed)
        val data = ByteArray(shape.volume) { 
            random.nextDouble(min, max).toInt().coerceIn(-128, 127).toByte()
        }
        return MockTensorInt8.fromArray(shape, data)
    }

    override fun randomUniform(shape: Shape, min: Double, max: Double, random: Random): Tensor<Int8, Byte> {
        val data = ByteArray(shape.volume) { 
            random.nextDouble(min, max).toInt().coerceIn(-128, 127).toByte()
        }
        return MockTensorInt8.fromArray(shape, data)
    }

    override fun fromArray(shape: Shape, data: FloatArray): Tensor<Int8, Byte> {
        val byteData = ByteArray(data.size) { data[it].toInt().coerceIn(-128, 127).toByte() }
        return MockTensorInt8.fromArray(shape, byteData)
    }

    override fun fromArray(shape: Shape, data: IntArray): Tensor<Int8, Byte> {
        val byteData = ByteArray(data.size) { data[it].coerceIn(-128, 127).toByte() }
        return MockTensorInt8.fromArray(shape, byteData)
    }
}

/**
 * Mock factories for creating tensors from byte data (file loading support).
 */

/**
 * Mock factory for creating FP32 tensors from byte data with real data but mock operations.
 */
public object MockFP32TensorFactory : TensorFactoryRegistry.TensorFromBytesFactory<FP32, Float> {
    
    override fun fromByteArray(shape: Shape, data: ByteArray, littleEndian: Boolean): Tensor<FP32, Float> {
        // Validate input data size matches shape requirements
        val expectedFloatCount = shape.volume
        val expectedByteSize = expectedFloatCount * 4 // 4 bytes per float
        
        require(data.size == expectedByteSize) {
            "Input data size (${data.size} bytes) does not match expected size for shape $shape " +
            "(expected $expectedByteSize bytes for $expectedFloatCount floats)"
        }
        
        // Convert byte array to float array using little-endian (GGUF standard)
        val floatArray = ByteArrayConverter.convertBytesToFloatArray(data, littleEndian = littleEndian)
        
        // Create and return the mock tensor using the existing factory method
        return MockTensorFP32.fromArray(shape, floatArray)
    }
}

/**
 * Mock factory for creating Int32 tensors from byte data with real data but mock operations.
 */
public object MockInt32TensorFactory : TensorFactoryRegistry.TensorFromBytesFactory<Int32, Int> {
    
    override fun fromByteArray(shape: Shape, data: ByteArray, littleEndian: Boolean): Tensor<Int32, Int> {
        // Validate input data size matches shape requirements
        val expectedIntCount = shape.volume
        val expectedByteSize = expectedIntCount * 4 // 4 bytes per int
        
        require(data.size == expectedByteSize) {
            "Input data size (${data.size} bytes) does not match expected size for shape $shape " +
            "(expected $expectedByteSize bytes for $expectedIntCount ints)"
        }
        
        // Convert byte array to int array
        val intArray = ByteArrayConverter.convertBytesToIntArray(data, littleEndian = littleEndian)
        
        // Create and return the mock tensor
        return MockTensorInt32.fromArray(shape, intArray)
    }
}

/**
 * Mock factory for creating Int8 tensors from byte data with real data but mock operations.
 */
public object MockInt8TensorFactory : TensorFactoryRegistry.TensorFromBytesFactory<Int8, Byte> {
    
    override fun fromByteArray(shape: Shape, data: ByteArray, littleEndian: Boolean): Tensor<Int8, Byte> {
        // Validate input data size matches shape requirements
        val expectedByteCount = shape.volume
        
        require(data.size == expectedByteCount) {
            "Input data size (${data.size} bytes) does not match expected size for shape $shape " +
            "(expected $expectedByteCount bytes)"
        }
        
        // Create and return the mock tensor directly from byte data
        return MockTensorInt8.fromArray(shape, data)
    }
}