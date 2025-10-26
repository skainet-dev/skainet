package sk.ainet.sk.ainet.exec.tensor.ops

import sk.ainet.context.ExecutionContext

/**
 * Re-export execute DSL in this package to be used by tests without extra imports.
 */
public inline fun <V> execute(ctx: ExecutionContext<V>, block: () -> Unit) {
    sk.ainet.context.execute(ctx, block)
}
