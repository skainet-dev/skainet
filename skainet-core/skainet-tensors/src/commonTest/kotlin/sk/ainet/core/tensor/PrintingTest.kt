package sk.ainet.core.tensor

import sk.ainet.core.tensor.backend.CpuTensorFP32
import sk.ainet.core.tensor.backend.CpuTensorInt8
import sk.ainet.core.tensor.backend.CpuTensorInt32
import kotlin.test.*

class PrintingTest {

    @Test
    fun testScalarPrinting() {
        // Test scalar (1D tensor with single element)
        val scalar = CpuTensorFP32.fromArray(Shape(1), floatArrayOf(42.5f))

        val scalarOutput = scalar.printScalar()
        assertEquals("42.5", scalarOutput)

        val generalOutput = scalar.print()
        assertEquals("42.5", generalOutput)
    }

    @Test
    fun testVectorPrinting() {
        // Test vector (1D tensor)
        val vector = CpuTensorFP32.fromArray(Shape(4), floatArrayOf(1.0f, 2.5f, -3.0f, 4.7f))

        val vectorOutput = vector.printVector()
        assertEquals("[1.0, 2.5, -3.0, 4.7]", vectorOutput)

        val generalOutput = vector.print()
        assertEquals("[1.0, 2.5, -3.0, 4.7]", generalOutput)
    }

    @Test
    fun testMatrixPrinting() {
        // Test matrix (2D tensor)
        val matrix = CpuTensorFP32.fromArray(
            Shape(2, 3),
            floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)
        )

        val matrixOutput = matrix.printMatrix()
        val expectedMatrix = """[
  [1.0, 2.0, 3.0],
  [4.0, 5.0, 6.0]
]"""
        assertEquals(expectedMatrix, matrixOutput)

        val generalOutput = matrix.print()
        assertEquals(expectedMatrix, generalOutput)
    }

    @Test
    fun testSingleElementMatrixPrinting() {
        // Test 1x1 matrix
        val singleMatrix = CpuTensorFP32.fromArray(Shape(1, 1), floatArrayOf(7.5f))

        val matrixOutput = singleMatrix.printMatrix()
        val expectedMatrix = """[
  [7.5]
]"""
        assertEquals(expectedMatrix, matrixOutput)

        val generalOutput = singleMatrix.print()
        assertEquals(expectedMatrix, generalOutput)
    }

    @Test
    fun testLargerMatrixPrinting() {
        // Test larger matrix
        val largerMatrix = CpuTensorFP32.fromArray(
            Shape(3, 2),
            floatArrayOf(1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f)
        )

        val matrixOutput = largerMatrix.printMatrix()
        val expectedMatrix = """[
  [1.1, 2.2],
  [3.3, 4.4],
  [5.5, 6.6]
]"""
        assertEquals(expectedMatrix, matrixOutput)
    }

    @Test
    fun testWrongRankForScalar() {
        // Test error handling for wrong tensor rank in printScalar
        val vector = CpuTensorFP32.fromArray(Shape(3), floatArrayOf(1.0f, 2.0f, 3.0f))

        assertFailsWith<IllegalArgumentException> {
            vector.printScalar()
        }
    }

    @Test
    fun testWrongRankForVector() {
        // Test error handling for wrong tensor rank in printVector
        val matrix = CpuTensorFP32.fromArray(Shape(2, 2), floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f))

        assertFailsWith<IllegalArgumentException> {
            matrix.printVector()
        }
    }

    @Test
    fun testWrongRankForMatrix() {
        // Test error handling for wrong tensor rank in printMatrix
        val vector = CpuTensorFP32.fromArray(Shape(3), floatArrayOf(1.0f, 2.0f, 3.0f))

        assertFailsWith<IllegalArgumentException> {
            vector.printMatrix()
        }
    }

    @Test
    fun test3DTensorPrinting() {
        // Test 3D tensor printing
        val tensor3D = CpuTensorFP32.fromArray(
            Shape(2, 2, 2),
            FloatArray(8) { it.toFloat() }
        )

        val output = tensor3D.print()
        val expected3D = """[
  [
    [0.0, 1.0],
    [2.0, 3.0]
  ],
  [
    [4.0, 5.0],
    [6.0, 7.0]
  ]
]"""
        assertEquals(expected3D, output)
        
        // Also test print3D directly
        val direct3D = tensor3D.print3D()
        assertEquals(expected3D, direct3D)
    }

    @Test
    fun testHigherDimensionTensorPrinting() {
        // Test that higher-dimensional tensors (4D+) show summary format
        val tensor4D = CpuTensorFP32.fromArray(
            Shape(2, 2, 2, 2),
            FloatArray(16) { it.toFloat() }
        )

        val output = tensor4D.print()
        assertTrue(output.contains("Tensor(shape=[2, 2, 2, 2], rank=4)"))
        assertTrue(output.contains("0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0"))
        assertTrue(output.contains("..."))
    }

    // Int8 Tensor Tests
    @Test
    fun testInt8ScalarPrinting() {
        // Test scalar (1D tensor with single element)
        val scalar = CpuTensorInt8.fromArray(Shape(1), byteArrayOf(42))

        val scalarOutput = scalar.printScalar()
        assertEquals("42", scalarOutput)

        val generalOutput = scalar.print()
        assertEquals("42", generalOutput)
    }

    @Test
    fun testInt8VectorPrinting() {
        // Test vector (1D tensor)
        val vector = CpuTensorInt8.fromArray(Shape(4), byteArrayOf(1, 25, -3, 47))

        val vectorOutput = vector.printVector()
        assertEquals("[1, 25, -3, 47]", vectorOutput)

        val generalOutput = vector.print()
        assertEquals("[1, 25, -3, 47]", generalOutput)
    }

    @Test
    fun testInt8MatrixPrinting() {
        // Test matrix (2D tensor)
        val matrix = CpuTensorInt8.fromArray(
            Shape(2, 3),
            byteArrayOf(1, 2, 3, 4, 5, 6)
        )

        val matrixOutput = matrix.printMatrix()
        val expectedMatrix = """[
  [1, 2, 3],
  [4, 5, 6]
]"""
        assertEquals(expectedMatrix, matrixOutput)

        val generalOutput = matrix.print()
        assertEquals(expectedMatrix, generalOutput)
    }

    @Test
    fun testInt8_3DTensorPrinting() {
        // Test 3D tensor printing
        val tensor3D = CpuTensorInt8.fromArray(
            Shape(2, 2, 2),
            ByteArray(8) { it.toByte() }
        )

        val output = tensor3D.print()
        val expected3D = """[
  [
    [0, 1],
    [2, 3]
  ],
  [
    [4, 5],
    [6, 7]
  ]
]"""
        assertEquals(expected3D, output)
        
        // Also test print3D directly
        val direct3D = tensor3D.print3D()
        assertEquals(expected3D, direct3D)
    }

    // Int32 Tensor Tests
    @Test
    fun testInt32ScalarPrinting() {
        // Test scalar (1D tensor with single element)
        val scalar = CpuTensorInt32.fromArray(Shape(1), intArrayOf(4200))

        val scalarOutput = scalar.printScalar()
        assertEquals("4200", scalarOutput)

        val generalOutput = scalar.print()
        assertEquals("4200", generalOutput)
    }

    @Test
    fun testInt32VectorPrinting() {
        // Test vector (1D tensor)
        val vector = CpuTensorInt32.fromArray(Shape(4), intArrayOf(100, 2500, -300, 4700))

        val vectorOutput = vector.printVector()
        assertEquals("[100, 2500, -300, 4700]", vectorOutput)

        val generalOutput = vector.print()
        assertEquals("[100, 2500, -300, 4700]", generalOutput)
    }

    @Test
    fun testInt32MatrixPrinting() {
        // Test matrix (2D tensor)
        val matrix = CpuTensorInt32.fromArray(
            Shape(2, 3),
            intArrayOf(100, 200, 300, 400, 500, 600)
        )

        val matrixOutput = matrix.printMatrix()
        val expectedMatrix = """[
  [100, 200, 300],
  [400, 500, 600]
]"""
        assertEquals(expectedMatrix, matrixOutput)

        val generalOutput = matrix.print()
        assertEquals(expectedMatrix, generalOutput)
    }

    @Test
    fun testInt32_3DTensorPrinting() {
        // Test 3D tensor printing
        val tensor3D = CpuTensorInt32.fromArray(
            Shape(2, 2, 2),
            IntArray(8) { it * 10 }
        )

        val output = tensor3D.print()
        val expected3D = """[
  [
    [0, 10],
    [20, 30]
  ],
  [
    [40, 50],
    [60, 70]
  ]
]"""
        assertEquals(expected3D, output)
        
        // Also test print3D directly
        val direct3D = tensor3D.print3D()
        assertEquals(expected3D, direct3D)
    }
}