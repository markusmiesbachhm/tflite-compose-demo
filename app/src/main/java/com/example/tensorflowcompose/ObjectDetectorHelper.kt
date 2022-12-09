package com.example.tensorflowcompose

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.google.android.gms.tflite.client.TfLiteInitializationOptions
import com.google.android.gms.tflite.gpu.support.TfLiteGpu
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.gms.vision.TfLiteVision
import org.tensorflow.lite.task.gms.vision.detector.Detection
import org.tensorflow.lite.task.gms.vision.detector.ObjectDetector

class ObjectDetectorHelper(
    private val context: Context,
    private val objectDetectorListener: DetectorListener
) {
    interface DetectorListener {
        fun onInitialized()
        fun onError(error: String)
        fun onResults(
            results: MutableList<Detection>?,
        )
    }

    private val TAG = "ObjectDetectionHelper"

    private val threshold = 0.5f
    private val numThreads = 2
    private val maxResults = 3
    private val modelName = "mobilenetv1.tflite"

    private lateinit var objectDetector: ObjectDetector
    private var gpuSupported = false

    init {
        TfLiteGpu.isGpuDelegateAvailable(context).onSuccessTask { gpuAvailable: Boolean ->
            val optionsBuilder =
                TfLiteInitializationOptions.builder()
            if (gpuAvailable) {
                optionsBuilder.setEnableGpuDelegateSupport(true)
                gpuSupported = true
            }
            TfLiteVision.initialize(context, optionsBuilder.build())
        }.addOnSuccessListener {
            setupObjectDetector()
            objectDetectorListener.onInitialized()
        }.addOnFailureListener {
            objectDetectorListener.onError(
                "TfLiteVision failed to initialize: "
                        + it.message
            )
        }
    }

    // Initialize the object detector using current settings on the thread that is using it.
    private fun setupObjectDetector() {
        // Set general detection options, including number of used threads
        val baseOptionsBuilder = BaseOptions.builder().setNumThreads(30)
        // Use the specified hardware for running the model. Default to CPU
        if (gpuSupported)
            baseOptionsBuilder.useGpu()

        // Create the base options for the detector using specifies max results and score threshold
        val objectDetectorOptionsBuilder =
            ObjectDetector.ObjectDetectorOptions.builder()
                .setScoreThreshold(threshold)
                .setMaxResults(maxResults)

        objectDetectorOptionsBuilder.setBaseOptions(baseOptionsBuilder.build())

        // Pre-built library for object detection
        objectDetector =
            ObjectDetector.createFromFileAndOptions(
                context,
                modelName,
                objectDetectorOptionsBuilder.build()
            )
    }

    fun detect(image: Bitmap, imageRotation: Int) {
        if (!TfLiteVision.isInitialized()) {
            Log.e(TAG, "detect: TfLiteVision is not initialized yet")
            return
        }

        // Create preprocessor for the image.
        val imageProcessor = ImageProcessor.Builder().add(Rot90Op(-imageRotation / 90)).build()

        // Preprocess the image and convert it into a TensorImage for detection.
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))

        val results = objectDetector.detect(tensorImage)
        objectDetectorListener.onResults(
            results
        )
    }
}

