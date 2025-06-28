package sk.ai.net.io.data.mnist

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals

/**
 * Tests for the MNIST data classes.
 */
class MNISTDataTest {

    /**
     * Tests the MNISTImage class.
     */
    @Test
    fun testMNISTImage() {
        // Create a test image
        val image = ByteArray(MNISTConstants.IMAGE_PIXELS) { it.toByte() }
        val label: Byte = 5
        val mnistImage = MNISTImage(image, label)
        
        // Verify the image
        assertEquals(image, mnistImage.image)
        assertEquals(label, mnistImage.label)
        
        // Test equals and hashCode
        val sameImage = MNISTImage(image, label)
        assertEquals(mnistImage, sameImage)
        assertEquals(mnistImage.hashCode(), sameImage.hashCode())
        
        // Test with different image
        val differentImage = ByteArray(MNISTConstants.IMAGE_PIXELS) { (it + 1).toByte() }
        val differentMnistImage1 = MNISTImage(differentImage, label)
        assertNotEquals(mnistImage, differentMnistImage1)
        assertNotEquals(mnistImage.hashCode(), differentMnistImage1.hashCode())
        
        // Test with different label
        val differentMnistImage2 = MNISTImage(image, 6)
        assertNotEquals(mnistImage, differentMnistImage2)
        assertNotEquals(mnistImage.hashCode(), differentMnistImage2.hashCode())
    }

    /**
     * Tests the MNISTDataset class.
     */
    @Test
    fun testMNISTDataset() {
        // Create test images
        val images = List(10) { i ->
            val image = ByteArray(MNISTConstants.IMAGE_PIXELS) { it.toByte() }
            val label = i.toByte()
            MNISTImage(image, label)
        }
        
        // Create a dataset
        val dataset = MNISTDataset(images)
        
        // Verify the dataset
        assertEquals(images, dataset.images)
        assertEquals(images.size, dataset.size)
        
        // Test subset
        val subset = dataset.subset(2, 5)
        assertEquals(3, subset.size)
        assertEquals(images.subList(2, 5), subset.images)
    }

    /**
     * Tests the MNISTLoaderConfig class.
     */
    @Test
    fun testMNISTLoaderConfig() {
        // Test default configuration
        val defaultConfig = MNISTLoaderConfig()
        assertEquals("mnist-data", defaultConfig.cacheDir)
        assertEquals(true, defaultConfig.useCache)
        
        // Test custom configuration
        val customConfig = MNISTLoaderConfig(
            cacheDir = "custom-cache-dir",
            useCache = false
        )
        assertEquals("custom-cache-dir", customConfig.cacheDir)
        assertEquals(false, customConfig.useCache)
    }

    /**
     * Tests the MNISTConstants object.
     */
    @Test
    fun testMNISTConstants() {
        // Verify the constants
        assertEquals(28, MNISTConstants.IMAGE_SIZE)
        assertEquals(28 * 28, MNISTConstants.IMAGE_PIXELS)
        
        assertEquals("train-images-idx3-ubyte.gz", MNISTConstants.TRAIN_IMAGES_FILENAME)
        assertEquals("train-labels-idx1-ubyte.gz", MNISTConstants.TRAIN_LABELS_FILENAME)
        assertEquals("t10k-images-idx3-ubyte.gz", MNISTConstants.TEST_IMAGES_FILENAME)
        assertEquals("t10k-labels-idx1-ubyte.gz", MNISTConstants.TEST_LABELS_FILENAME)
        
        assertEquals("https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz", MNISTConstants.TRAIN_IMAGES_URL)
        assertEquals("https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz", MNISTConstants.TRAIN_LABELS_URL)
        assertEquals("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz", MNISTConstants.TEST_IMAGES_URL)
        assertEquals("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz", MNISTConstants.TEST_LABELS_URL)
    }
}