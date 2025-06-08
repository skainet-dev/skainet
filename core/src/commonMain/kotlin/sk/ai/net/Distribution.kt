package sk.ai.net


interface Distribution<T> {

    /**
     * Seed for the random number generator.
     */
    val seed: Long

    /**
     * Retrieves a new sample from the distribution.
     *
     * @return A T value.
     */
    fun sample(): T


    /**
     * Retrieves the distribution's mean.
     *
     * @return The mean value.
     */
    val mean: T

    /**
     * Retrieves the distribution's standard deviation.
     *
     * @return The standard deviation.
     */
    val deviation: T
}
