package sk.ainet.lang.nn.topology

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType


public sealed class ModuleParameter<T : DType, V> {
    public abstract val name: String
    public abstract var value: Tensor<T, V>

    public data class WeightParameter<T : DType, V>(
        override val name: String,
        override var value: Tensor<T, V>
    ) : ModuleParameter<T, V>()

    public data class BiasParameter<T : DType, V>(
        override val name: String,
        override var value: Tensor<T, V>
    ) : ModuleParameter<T, V>()
}

public interface ModuleParameters<T : DType, V> {
    public val params: List<ModuleParameter<T, V>>
}

public fun <T : DType, V> List<ModuleParameter<T, V>>.by(name: String): ModuleParameter<T, V>? =
    firstOrNull { namedParameter -> namedParameter.name.uppercase().contains(name.uppercase()) }

// Returns the first BiasParameter or throws a NoSuchElementException if none is found.
public fun <T : DType, V> List<ModuleParameter<T, V>>.bias(): ModuleParameter.BiasParameter<T, V> =
    this.filterIsInstance<ModuleParameter.BiasParameter<T, V>>()
        .firstOrNull() ?: throw NoSuchElementException("No bias parameter found!")

// Returns the first WeightParameter or throws a NoSuchElementException if none is found.
public fun <T : DType, V> List<ModuleParameter<T, V>>.weights(): ModuleParameter.WeightParameter<T, V> =
    this.filterIsInstance<ModuleParameter.WeightParameter<T, V>>()
        .firstOrNull() ?: throw NoSuchElementException("No weight parameter found!")
