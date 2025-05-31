package sk.ai.net.io.data.mnist;



import io.ktor.client.HttpClient
import io.ktor.client.engine.cio.CIO
import io.ktor.client.plugins.logging.Logging
import io.ktor.client.request.get
import io.ktor.client.request.headers
import io.ktor.utils.io.jvm.javaio.copyTo
import io.ktor.utils.io.core.use
import io.ktor.client.call.body
import io.ktor.client.plugins.onDownload
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.client.request.get
import io.ktor.client.statement.bodyAsChannel
import io.ktor.utils.io.copyAndClose
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.util.zip.GZIPInputStream

import kotlinx.coroutines.runBlocking
import java.nio.file.Files
import java.nio.file.Paths
import java.util.zip.GZIPOutputStream






import io.ktor.client.statement.HttpResponse

suspend fun downloadFile(urlStr: String, outputPath: String) {
    val client = HttpClient(CIO) {
        install(Logging)
    }
    // This is the correct way to initialize the HttpClient with CIO
    try {
        val file = File(outputPath)

        val httpClient = HttpClient {
            install(Logging)
        }

        val httpResponse: HttpResponse = client.get(urlStr) {
            onDownload { bytesSentTotal, contentLength ->
                println("Received $bytesSentTotal bytes from $contentLength")
            }
        }
        val responseBody: ByteArray = httpResponse.body()
        file.writeBytes(responseBody)
        println("A file saved to ${file.path}")

    } finally {
        client.close()
    }
}




fun decompressGzipFile(gzipFilePath: String, outputFilePath: String) {
    GZIPInputStream(FileInputStream(gzipFilePath)).use { gzipInputStream ->
        FileOutputStream(outputFilePath).use { outputStream ->
            val buffer = ByteArray(1024)
            var len: Int
            while (gzipInputStream.read(buffer).also { len = it } > 0) {
                outputStream.write(buffer, 0, len)
            }
        }
    }
    println("Decompressed $gzipFilePath to $outputFilePath")
}


fun downloadMnistDataset() = runBlocking {
    val urls = mapOf(
        "train-images-idx3-ubyte.gz" to "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz" to "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz" to "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz" to "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"
    )

    // https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz

    urls.forEach { (fileName, url) ->
        val outputPath = Paths.get(fileName).toString()
        if (!Files.exists(Paths.get(outputPath))) {
            downloadFile(url, outputPath)
        } else {
            println("$fileName already exists. Skipping download.")
        }
    }
}

fun decompress() {
    val gzipFiles = listOf(
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    )

    gzipFiles.forEach { gzipFile ->
        val outputFile = gzipFile.removeSuffix(".gz")
        decompressGzipFile(gzipFile, outputFile)
    }
}


