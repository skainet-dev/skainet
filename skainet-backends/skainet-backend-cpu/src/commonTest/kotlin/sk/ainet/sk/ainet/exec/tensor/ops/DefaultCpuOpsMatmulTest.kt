package sk.ainet.sk.ainet.exec.tensor.ops

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.tensor.matmul
import sk.ainet.lang.tensor.pprint

class DefaultCpuOpsMatmulTest {
    private val dataFactory = DenseTensorDataFactory()
    private val cpuOpsF = DefaultCpuOps<Float>(dataFactory)
    private val cpuOpsI = DefaultCpuOps<Int>(dataFactory)

    private fun fTensor(shape: Shape, values: FloatArray): VoidOpsTensor<FP32, Float> {
        val data = dataFactory.fromFloatArray<FP32, Float>(shape, FP32::class, values)
        return VoidOpsTensor(data, FP32::class)
    }

    @Test
    fun matmul_fp32_2d() {
        // 2x3 @ 3x2 = 2x2
        val a = fTensor(Shape(2, 3), floatArrayOf(
            1f, 2f, 3f,
            4f, 5f, 6f
        ))
        val b = fTensor(Shape(3, 2), floatArrayOf(
            7f, 8f,
            9f, 10f,
            11f, 12f
        ))
        val r = cpuOpsF.matmul(a, b)
        assertEquals(Shape(2, 2), r.shape)
        // Manual result
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assertEquals(58f, r.data[0, 0])
        assertEquals(64f, r.data[0, 1])
        assertEquals(139f, r.data[1, 0])
        assertEquals(154f, r.data[1, 1])
    }

    @Test
    fun matmul_fp32_batched3d() {
        // batch=2, (2, 3, 4) @ (2, 4, 2) -> (2, 3, 2)
        val a = fTensor(Shape(2, 3, 4), FloatArray(2*3*4) { it.toFloat() + 1 })
        val b = fTensor(Shape(2, 4, 2), FloatArray(2*4*2) { (it % 5).toFloat() + 1 })
        val r = cpuOpsF.matmul(a, b)
        assertEquals(Shape(2, 3, 2), r.shape)
        // spot-check a couple values by recomputing
        fun ref(batch: Int, m: Int, n: Int): Float {
            var acc = 0f
            val k = 4
            for (i in 0 until k) {
                val av = a.data[batch, m, i] as Float
                val bv = b.data[batch, i, n] as Float
                acc += av * bv
            }
            return acc
        }
        assertEquals(ref(0, 0, 0), r.data[0, 0, 0])
        assertEquals(ref(1, 2, 1), r.data[1, 2, 1])
    }

    @Test
    fun matmul_fp32_bchw_images_last2dims() {
        // Treat last two dims as (H,W) x (W,N) per BCH batch/channel
        val B = 2; val C = 3; val H = 2; val W = 3; val N = 2
        val aShape = Shape(B, C, H, W)
        val bShape = Shape(B, C, W, N) // broadcast over B,C exact match
        val a = fTensor(aShape, FloatArray(B*C*H*W) { (it + 1).toFloat() })
        val b = fTensor(bShape, FloatArray(B*C*W*N) { ((it % 7) + 1).toFloat() })

        val r = cpuOpsF.matmul(a, b)
        assertEquals(Shape(B, C, H, N), r.shape)
        // Validate per-slice via zero-copy slicing semantics by direct indexing
        fun ref(bi: Int, ci: Int, h: Int, n: Int): Float {
            var acc = 0f
            for (w in 0 until W) {
                val av = a.data[bi, ci, h, w] as Float
                val bv = b.data[bi, ci, w, n] as Float
                acc += av * bv
            }
            return acc
        }
        assertEquals(ref(0, 0, 0, 0), r.data[0, 0, 0, 0])
        assertEquals(ref(1, 2, 1, 1), r.data[1, 2, 1, 1])
    }

    @Test
    fun matmul_mismatched_inner_dims_throws() {
        val a = fTensor(Shape(2, 3), FloatArray(6) { 1f })
        val b = fTensor(Shape(4, 2), FloatArray(8) { 1f })
        assertFailsWith<IllegalArgumentException> { cpuOpsF.matmul(a, b) }
    }

    @Test
    fun matmul_zero_sizes() {
        val a = fTensor(Shape(2, 0), FloatArray(0))
        val b = fTensor(Shape(0, 3), FloatArray(0))
        val r = cpuOpsF.matmul(a, b)
        assertEquals(Shape(2, 3), r.shape)
        // All zeros by definition of sum over empty set
        assertEquals(0, r.shape.volume)
    }

    @Test
    fun matmul_exec_context_extension() {
        val ctx = DirectCpuExecutionContext<Float>()
        execute(ctx) {
            val a = tensor<FP32, Float> {
                shape(2, 3) {
                    init { idx -> (idx[0]*3 + idx[1] + 1).toFloat() }
                }
            }
            val b = tensor<FP32, Float> {
                shape(3, 2) {
                    init { idx -> (idx[0]*2 + idx[1] + 1).toFloat() }
                }
            }
            val r = a.matmul(b)
            // Check a couple known values
            assertEquals(58f, r.data[0, 0])
            assertEquals(64f, r.data[0, 1])
        }
    }
}
