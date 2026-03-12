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
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
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
        private const val DROWSY_THRESHOLD = 0.5f   // above 50% = drowsy
        private const val ALERT_FRAMES     = 2       // out of rolling window below
        private const val WINDOW_SIZE      = 4       // rolling window size
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
    private var isAlertActive           = false   // tracks drowsy alert state
    private val predictionWindow        = ArrayDeque<Boolean>(WINDOW_SIZE) // rolling window
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
        isAlertActive    = false
        btnStart.text    = getString(R.string.start_detection)
        tvStatus.text    = "Detection stopped — camera preview active"
        tvAlert.visibility   = View.GONE
        tvConfidence.text    = "Confidence: --"
        predictionWindow.clear()
    }

    // ── Inference ────────────────────────────────────────────────────────────
    private fun runInference(imageProxy: ImageProxy) {
        val tfl = tflite ?: run { imageProxy.close(); return }
        try {
            val raw = imageProxy.toBitmap()
            imageProxy.close()

            // Scale to model input size
            val bitmap = Bitmap.createScaledBitmap(raw, INPUT_SIZE, INPUT_SIZE, true)

            // Build float32 ByteBuffer: [1, H, W, 3] normalized to [0, 1]
            val inputBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
            inputBuffer.order(ByteOrder.nativeOrder())
            inputBuffer.rewind()  // ← CRITICAL: must be at position 0 before filling
            val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
            bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
            for (pixel in pixels) {
                inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 127.5f - 1.0f) // R → [-1, 1]
                inputBuffer.putFloat(((pixel shr 8)  and 0xFF) / 127.5f - 1.0f) // G → [-1, 1]
                inputBuffer.putFloat((pixel           and 0xFF) / 127.5f - 1.0f) // B → [-1, 1]
            }
            inputBuffer.rewind()  // ← CRITICAL: reset to 0 before passing to TFLite

            // Detect output shape at runtime
            val outputShape = tfl.getOutputTensor(0).shape()
            val numClasses = if (outputShape.size >= 2) outputShape[outputShape.size - 1] else 1

            val score0: Float
            val score1: Float
            if (numClasses >= 2) {
                val output = Array(1) { FloatArray(numClasses) }
                tfl.run(inputBuffer, output)
                score0 = output[0][0]
                score1 = output[0][1]
            } else {
                val output = Array(1) { FloatArray(1) }
                tfl.run(inputBuffer, output)
                score0 = output[0][0]
                score1 = output[0][0]
            }

            // Log both classes so we can tell which index is drowsy
            Log.d(TAG, "scores → [0]=${"%05.3f".format(score0)}  [1]=${"%05.3f".format(score1)}")

            // Model class order: [Drowsy=0, Non Drowsy=1]
            val drowsyScore = score0
            val awakeScore  = 1f - drowsyScore  // always sums to 100%

            // Rolling window majority vote
            if (predictionWindow.size >= WINDOW_SIZE) predictionWindow.removeFirst()
            predictionWindow.addLast(drowsyScore >= DROWSY_THRESHOLD)
            val drowsyVotes = 20*predictionWindow.count { it }
            val alert = drowsyVotes >= ALERT_FRAMES

            runOnUiThread { updateUI(drowsyScore, awakeScore, alert) }
        } catch (e: Exception) {
            Log.e(TAG, "Inference error: ${e.message}")
            imageProxy.close()
        }
    }

    private fun updateUI(drowsyScore: Float, awakeScore: Float, alert: Boolean) {
        val drowsyPct = (drowsyScore * 100).toInt()
        val awakePct  = (awakeScore  * 100).toInt()
        tvConfidence.text = "Drowsy: $drowsyPct%  |  Awake: $awakePct%"

        if (alert && !isAlertActive) {
            // Transition: awake → drowsy — fire alert once
            isAlertActive = true
            tvStatus.text      = "😴 DROWSY"
            tvAlert.visibility = View.VISIBLE
            toneGenerator?.startTone(ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD, 1500)
        } else if (!alert && isAlertActive) {
            // Transition: drowsy → awake — clear alert
            isAlertActive = false
            tvStatus.text      = "✅ AWAKE"
            tvAlert.visibility = View.GONE
        } else if (!alert) {
            tvStatus.text      = "✅ AWAKE"
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
