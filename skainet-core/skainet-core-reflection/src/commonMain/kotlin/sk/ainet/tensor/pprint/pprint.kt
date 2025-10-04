package sk.ainet.tensor.pprint

import sk.ainet.core.tensor.DType
import sk.ainet.core.tensor.Tensor

public fun <T: DType,V> Tensor<T,V>.pprint(): String {
    return when (this.shape.rank) {
        0 -> {
            // Scalar - single value
            this.toString()
        }
        1 -> {
            // Vector - horizontal representation with parentheses
            val sb = StringBuilder()
            sb.append("( ")
            for (i in 0 until this.shape[0]) {
                if (i > 0) sb.append(", ")
                sb.append(this[i].toString())
            }
            sb.append(" )")
            sb.toString()
        }
        2 -> {
            // Matrix - vertical representation with Unicode brackets
            val sb = StringBuilder()
            val rows = this.shape[0]
            val cols = this.shape[1]
            
            for (i in 0 until rows) {
                // Left bracket
                val leftBracket = when {
                    rows == 1 -> "( "
                    i == 0 -> "⎛ "
                    i == rows - 1 -> "⎝ "
                    else -> "⎜ "
                }
                sb.append(leftBracket)
                
                // Matrix elements
                for (j in 0 until cols) {
                    if (j > 0) sb.append(", ")
                    sb.append(this[i, j].toString())
                }
                
                // Right bracket
                val rightBracket = when {
                    rows == 1 -> " )"
                    i == 0 -> " ⎞"
                    i == rows - 1 -> " ⎠"
                    else -> " ⎟"
                }
                sb.append(rightBracket)
                
                if (i < rows - 1) sb.append("\n")
            }
            sb.toString()
        }
        else -> {
            // Higher rank tensors - use toString
            this.toString()
        }
    }
}