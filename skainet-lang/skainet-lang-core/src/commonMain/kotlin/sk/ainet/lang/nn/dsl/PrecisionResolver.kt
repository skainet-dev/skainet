package sk.ainet.lang.nn.dsl

import sk.ainet.lang.types.DType

public interface PrecisionResolver<V> {
    public fun <TNetwork : DType, TBlock : DType, TLayer : DType> resolvePrecision(
        networkType: TNetwork,
        blockType: TBlock?,
        layerType: TLayer?
    ): DType
}