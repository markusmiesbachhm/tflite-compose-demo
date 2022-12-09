package com.example.tensorflowcompose

import android.app.Application
import android.util.Log
import androidx.compose.runtime.State
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModel
import org.tensorflow.lite.task.gms.vision.detector.Detection

private const val TAG = "MainViewModel"

class MainViewModel(application: Application) : AndroidViewModel(application), ObjectDetectorHelper.DetectorListener {

    private val _isInitialized = mutableStateOf(false)
    val isInitialized: State<Boolean> = _isInitialized

    private val _results = mutableStateOf<List<Detection>>(listOf())
    val results: State<List<Detection>> = _results

    override fun onInitialized() {
        _isInitialized.value = true
    }

    override fun onError(error: String) {
        Log.e(TAG, error)
    }

    override fun onResults(
        results: MutableList<Detection>?,
    ) {
        _results.value = results ?: listOf()
    }
}