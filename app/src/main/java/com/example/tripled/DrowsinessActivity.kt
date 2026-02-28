package com.example.tripled

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class DrowsinessActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "DrowsinessDetection"

        /**
         * Place your trained TFLite model in:
         *   app/src/main/assets/drowsiness_model.tflite
         *
         * Input  : [1, INPUT_SIZE, INPUT_SIZE, 3]  float32 RGB
         * Output : [1, 2]  [awake_score, drowsy_score]
         *       OR [1, 1]  drowsiness probability (change output[0][1] → output[0][0])
         */
        private const val MODEL_NAME       = "drowsiness_model.tflite"
        private const val INPUT_SIZE       = 224
        private const val DROWSY_THRESHOLD = 0.6f
        private const val ALERT_FRAMES     = 5
    }

    private lateinit var previewView    : PreviewView
    private lateinit var tvStatus       : TextView
    private lateinit var tvAlert        : TextView
    private lateinit var tvConfidence   : TextView
    private lateinit var btnStart       : Button

    private lateinit var cameraExecutor : ExecutorService
    private var cameraProvider          : ProcessCameraProvider? = null
    private var imageAnalysis           : ImageAnalysis? = null
    private var tflite                  : Interpreter? = null
    private var isDetecting             = false
    private var drowsyFrameCount        = 0
    private var modelLoaded             = false
    private var toneGenerator           : ToneGenerator? = null

    // ── Permission launcher ──────────────────────────────────────────────────
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
        else {
            Toast.makeText(this, "Camera permission is required.", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_drowsiness)

        previewView  = findViewById(R.id.cameraPreview)
        tvStatus     = findViewById(R.id.tvStatus)
        tvAlert      = findViewById(R.id.tvAlert)
        tvConfidence = findViewById(R.id.tvConfidence)
        btnStart     = findViewById(R.id.btnStartDetection)

        cameraExecutor = Executors.newSingleThreadExecutor()
        toneGenerator  = ToneGenerator(AudioManager.STREAM_ALARM, 100)

        // Load model (camera still starts regardless)
        modelLoaded = loadModel()

        // Camera starts immediately — no button press required
        requestCameraPermission()

        btnStart.setOnClickListener {
            if (!modelLoaded) {
                Toast.makeText(this,
                    "Add $MODEL_NAME to app/src/main/assets/ first",
                    Toast.LENGTH_LONG).show()
                return@setOnClickListener
            }
            if (!isDetecting) startDetection() else stopDetection()
        }
    }

    // ── Model loading ────────────────────────────────────────────────────────
    private fun loadModel(): Boolean {
        return try {
            tflite = Interpreter(loadModelFile())
            Log.d(TAG, "TFLite model loaded.")
            true
        } catch (e: Exception) {
            Log.w(TAG, "No model found: ${e.message}")
            false
        }
    }

    private fun loadModelFile(): MappedByteBuffer {
        val afd = assets.openFd(MODEL_NAME)
        return FileInputStream(afd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
    }

    // ── Camera ───────────────────────────────────────────────────────────────
    private fun requestCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) startCamera()
        else permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            cameraProvider = future.get()
            bindCamera()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCamera() {
        val provider = cameraProvider ?: return

        val selector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
            .build()

        val preview = Preview.Builder().build().also {
            it.surfaceProvider = previewView.surfaceProvider
        }

        // ImageAnalysis use case — only attached when detection starts
        imageAnalysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        try {
            provider.unbindAll()
            // Bind preview only first; analysis added when Start is pressed
            provider.bindToLifecycle(this, selector, preview)

            val modelStatus = if (modelLoaded) "Model loaded ✓" else "⚠ No model — preview only"
            tvStatus.text = modelStatus
        } catch (e: Exception) {
            Log.e(TAG, "Camera bind failed: ${e.message}")
            tvStatus.text = "❌ Camera error: ${e.message}"
        }
    }

    // ── Detection start / stop ───────────────────────────────────────────────
    private fun startDetection() {
        val provider = cameraProvider ?: return
        val analysis = imageAnalysis ?: return

        val selector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
            .build()

        val preview = Preview.Builder().build().also {
            it.surfaceProvider = previewView.surfaceProvider
        }

        analysis.setAnalyzer(cameraExecutor) { runInference(it) }

        try {
            provider.unbindAll()
            provider.bindToLifecycle(this, selector, preview, analysis)
            isDetecting = true
            btnStart.text = "⏹  Stop Detection"
            tvStatus.text = "🔍 Detecting…"
        } catch (e: Exception) {
            Log.e(TAG, "Detection bind failed: ${e.message}")
        }
    }

    private fun stopDetection() {
        imageAnalysis?.clearAnalyzer()
        // Re-bind preview only (keep camera visible)
        cameraProvider?.let { provider ->
            val selector = CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_FRONT).build()
            val preview = Preview.Builder().build().also {
                it.surfaceProvider = previewView.surfaceProvider
            }
            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, selector, preview)
            } catch (e: Exception) { Log.e(TAG, "Rebind failed: ${e.message}") }
        }
        isDetecting      = false
        drowsyFrameCount = 0
        btnStart.text    = getString(R.string.start_detection)
        tvStatus.text    = "Detection stopped — camera preview active"
        tvAlert.visibility   = View.GONE
        tvConfidence.text    = "Confidence: --"
    }

    // ── Inference ────────────────────────────────────────────────────────────
    private fun runInference(imageProxy: ImageProxy) {
        val tfl = tflite ?: run { imageProxy.close(); return }
        try {
            val bitmap = imageProxy.toBitmap()
            imageProxy.close()

            val processor = ImageProcessor.Builder()
                .add(ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                .build()
            val tensorImage = processor.process(TensorImage.fromBitmap(bitmap))

            // Adjust output shape to match your model: [1,2] two-class or [1,1] single
            val output = Array(1) { FloatArray(2) }
            tfl.run(tensorImage.buffer, output)

            val drowsyScore = output[0][1]   // index [1] = drowsy class
            if (drowsyScore >= DROWSY_THRESHOLD) drowsyFrameCount++ else drowsyFrameCount = 0

            runOnUiThread { updateUI(drowsyScore, drowsyFrameCount >= ALERT_FRAMES) }
        } catch (e: Exception) {
            Log.e(TAG, "Inference error: ${e.message}")
            imageProxy.close()
        }
    }

    private fun updateUI(score: Float, alert: Boolean) {
        tvConfidence.text = "Drowsy confidence: ${"%.1f".format(score * 100)}%"
        if (alert) {
            tvStatus.text      = "😴 DROWSY"
            tvAlert.visibility = View.VISIBLE
            toneGenerator?.startTone(ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD, 500)
        } else {
            tvStatus.text      = "✅ AWAKE"
            tvAlert.visibility = View.GONE
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    override fun onDestroy() {
        super.onDestroy()
        cameraProvider?.unbindAll()
        cameraExecutor.shutdown()
        tflite?.close()
        toneGenerator?.release()
    }
}
