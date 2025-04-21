package sk.ai.net.safetensors

interface BiMap<K : Any, V : Any> : Map<K, V> {
    override val values: Set<V>
    val inverse: BiMap<V, K>
}

interface MutableBiMap<K : Any, V : Any> : BiMap<K, V>, MutableMap<K, V> {
    override val values: MutableSet<V>
    override val inverse: MutableBiMap<V, K>

    fun forcePut(key: K, value: V): V?
}