package com.example.tensorflowcompose

import android.Manifest
import android.content.ContentValues.TAG
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Camera
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Surface
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.tensorflowcompose.ui.theme.TensorFlowComposeTheme
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private val TAG = "MainActivity"

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            Log.i(TAG, "Permission granted")
        } else {
            Log.i(TAG, "Permission denied")
        }
    }

    private fun requestCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                Log.i(TAG, "Permission previously granted")
            }

            ActivityCompat.shouldShowRequestPermissionRationale(
                this,
                Manifest.permission.CAMERA
            ) -> Log.i(TAG, "Show camera permissions dialog")

            else -> requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            val mainViewModel: MainViewModel by viewModels()
            TensorFlowComposeTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colors.background
                ) {
                    Column() {
                        ComposableCamera(
                            modifier = Modifier.weight(0.8f, true),
                            viewModel = mainViewModel
                        )
                        Results(modifier = Modifier.weight(0.2f, true), viewModel = mainViewModel)
                    }
                }
            }
        }
        requestCameraPermission()
    }
}

@Composable
fun Results(modifier: Modifier, viewModel: MainViewModel) {
    LazyColumn(modifier = modifier.fillMaxWidth()) {
        items(viewModel.results.value) {
            Text(text = it.categories[0].label + " " + it.categories[0].score)
        }
    }
}

@Composable
fun ComposableCamera(modifier: Modifier, viewModel: MainViewModel) {
    val lifecycleOwner = LocalLifecycleOwner.current
    val context = LocalContext.current

    val cameraProviderFuture = remember {
        ProcessCameraProvider.getInstance(context)
    }

    val previewView = remember {
        PreviewView(context).apply {
            id = R.id.preview_view
        }
    }

    val objectDetectorHelper = remember { ObjectDetectorHelper(context, viewModel) }

    if (viewModel.isInitialized.value) {
        val cameraExecutor = remember {
            Executors.newSingleThreadExecutor()
        }

        AndroidView(factory = { previewView }, modifier = modifier.fillMaxWidth()) {
            cameraProviderFuture.addListener({
                val cameraProvider = cameraProviderFuture.get()
                // Preview. Only using the 4:3 ratio because this is the closest to the tensorflow models
                val preview = androidx.camera.core.Preview.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                    .build()
                    .also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                // ImageAnalysis. Using RGBA 8888 to match how tensorflow models work
                val imageAnalyzer = ImageAnalysis.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                    .also {
                        it.setAnalyzer(cameraExecutor) { image ->
                            val bitmapBuffer = Bitmap.createBitmap(
                                image.width,
                                image.height,
                                Bitmap.Config.ARGB_8888
                            )
                            image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }

                            val imageRotation = image.imageInfo.rotationDegrees
                            // Pass Bitmap and rotation to the object detector helper for processing and detection
                            objectDetectorHelper.detect(bitmapBuffer, imageRotation)
                        }
                    }

                try {
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        cameraSelector,
                        preview,
                        imageAnalyzer
                    )
                } catch (e: Exception) {
                    Log.e(TAG, "CameraX ${e.localizedMessage}")
                }
            }, ContextCompat.getMainExecutor(context))
        }
    }
}
