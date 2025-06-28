package sk.ai.net.io.data.mnist

import kotlinx.serialization.Serializable

/**
 * Represents a single MNIST image with its label.
 *
 * @property image The pixel data of the image as a ByteArray (28x28 pixels).
 * @property label The label of the image (0-9).
 */
@Serializable
data class MNISTImage(
    val image: ByteArray,
    val label: Byte
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as MNISTImage

        if (!image.contentEquals(other.image)) return false
        if (label != other.label) return false

        return true
    }

    override fun hashCode(): Int {
        var result = image.contentHashCode()
        result = 31 * result + label.toInt()
        return result
    }
}

/**
 * Represents a dataset of MNIST images.
 *
 * @property images The list of MNIST images.
 */
@Serializable
data class MNISTDataset(
    val images: List<MNISTImage>
) {
    /**
     * Returns the number of images in the dataset.
     */
    val size: Int
        get() = images.size

    /**
     * Returns a subset of the dataset.
     *
     * @param fromIndex The starting index (inclusive).
     * @param toIndex The ending index (exclusive).
     * @return A new MNISTDataset containing the specified range of images.
     */
    fun subset(fromIndex: Int, toIndex: Int): MNISTDataset {
        return MNISTDataset(images.subList(fromIndex, toIndex))
    }
}

/**
 * Configuration for the MNIST loader.
 *
 * @property cacheDir The directory where downloaded files will be cached.
 * @property useCache Whether to use cached files if available.
 */
data class MNISTLoaderConfig(
    val cacheDir: String = "mnist-data",
    val useCache: Boolean = true
)

/**
 * Constants for the MNIST dataset.
 */
object MNISTConstants {
    const val IMAGE_SIZE = 28
    const val IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
    
    const val TRAIN_IMAGES_FILENAME = "train-images-idx3-ubyte.gz"
    const val TRAIN_LABELS_FILENAME = "train-labels-idx1-ubyte.gz"
    const val TEST_IMAGES_FILENAME = "t10k-images-idx3-ubyte.gz"
    const val TEST_LABELS_FILENAME = "t10k-labels-idx1-ubyte.gz"
    
    const val TRAIN_IMAGES_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz"
    const val TRAIN_LABELS_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz"
    const val TEST_IMAGES_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz"
    const val TEST_LABELS_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"
}

/**
 * Interface for the MNIST loader.
 */
interface MNISTLoader {
    /**
     * Loads the MNIST training dataset.
     *
     * @return The MNIST training dataset.
     */
    suspend fun loadTrainingData(): MNISTDataset
    
    /**
     * Loads the MNIST test dataset.
     *
     * @return The MNIST test dataset.
     */
    suspend fun loadTestData(): MNISTDataset
}