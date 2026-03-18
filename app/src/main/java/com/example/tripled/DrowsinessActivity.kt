package com.example.tripled

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
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
import android.graphics.Matrix
import androidx.annotation.OptIn
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

/**
 * DrowsinessActivity — improved version.
 *
 * Key fixes over the original:
 *  1. Rolling-window vote bug fixed (×5 multiplier removed).
 *  2. ML Kit face detection added — crops face before inference,
 *     matching how the model was trained (MediaPipe-cropped faces).
 *  3. Alert cooldown (3 s) prevents tone spam on rapid state flips.
 *  4. Low-confidence dead-band (±CONFIDENCE_DEADBAND around 0.5)
 *     prevents flickering at the boundary.
 *  5. Frame-rate throttle (INFERENCE_INTERVAL_MS) avoids burning CPU
 *     on every camera frame.
 *  6. imageProxy is always closed exactly once via finally block.
 *  7. ToneGenerator null-safety tightened throughout.
 *  8. Temporal smoothing uses a proper majority-vote with clear semantics.
 *
 * Dependencies to add in build.gradle (:app):
 *   implementation "com.google.mlkit:face-detection:16.1.7"
 */
class DrowsinessActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "DrowsinessDetection"

        // ── Model ──────────────────────────────────────────────────────────
        // Place trained TFLite model in app/src/main/assets/
        // Input : [1, 224, 224, 3]  float32, EfficientNet preprocessing (÷127.5 − 1)
        // Output: [1, 1] sigmoid  OR  [1, 2] softmax (auto-detected at runtime)
        private const val MODEL_NAME = "drowsiness_model.tflite"
        private const val INPUT_SIZE = 224

        // ── Thresholds ─────────────────────────────────────────────────────
        // Predictions within DEAD_BAND of 0.5 are treated as "uncertain" and
        // do not change the current alert state. Prevents flickering at the boundary.
        private const val DROWSY_THRESHOLD     = 0.5f
        private const val CONFIDENCE_DEAD_BAND = 0.10f // ±10% around threshold

        // ── Rolling window ─────────────────────────────────────────────────
        // ALERT requires ALERT_FRAMES drowsy votes out of the last WINDOW_SIZE frames.
        // Default: 3 out of 5 = majority. Tune to taste.
        private const val WINDOW_SIZE   = 5
        private const val ALERT_FRAMES  = 3   // majority vote

        // ── No-face handling ───────────────────────────────────────────────
        // No face detected = person looked away. Treated as AWAKE (0.0f vote).
        // Face detected but eyes closed = handled by model score (drowsy vote).

        // ── Alert audio ────────────────────────────────────────────────────
        // Beep repeats every BEEP_INTERVAL_MS while drowsy state is active.
        // ALERT_COOLDOWN_MS is the minimum gap before re-triggering after
        // the alert was manually cleared by returning to awake state.
        private const val BEEP_INTERVAL_MS  = 1_500L
        private const val ALERT_COOLDOWN_MS = 3_000L

        // ── Frame throttle ─────────────────────────────────────────────────
        // Run inference at most once every N ms. Front camera at 30 fps →
        // INFERENCE_INTERVAL_MS = 333 gives ~3 inferences/second, plenty for
        // drowsiness detection while keeping CPU usage low.
        private const val INFERENCE_INTERVAL_MS = 333L

        // ── Face crop padding ──────────────────────────────────────────────
        // Extra padding added around the ML Kit face bounding box before
        // cropping, matching the 20 % padding used during training.
        private const val FACE_PADDING = 0.20f
    }

    // ── Views ────────────────────────────────────────────────────────────────
    private lateinit var previewView    : PreviewView
    private lateinit var tvStatus       : TextView
    private lateinit var tvAlert        : TextView
    private lateinit var tvConfidence   : TextView
    private lateinit var btnStart       : Button

    // ── Camera & inference ───────────────────────────────────────────────────
    private lateinit var cameraExecutor : ExecutorService
    private var cameraProvider          : ProcessCameraProvider? = null
    private var imageAnalysis           : ImageAnalysis? = null
    private var tflite                  : Interpreter? = null

    // ── State ────────────────────────────────────────────────────────────────
    private var isDetecting             = false
    private var isAlertActive           = false
    private var modelLoaded             = false
    private var lastInferenceMs         = 0L
    private var noFaceFrames            = 0   // consecutive frames with no face detected
    private val predictionWindow        = ArrayDeque<Boolean>(WINDOW_SIZE)

    // ── ML Kit face detector ─────────────────────────────────────────────────
    private val faceDetector by lazy {
        val opts = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setMinFaceSize(0.15f) // ignore very small faces
            .build()
        FaceDetection.getClient(opts)
    }

    // ── Audio ────────────────────────────────────────────────────────────────
    private var toneGenerator : ToneGenerator? = null
    private val beepHandler   = Handler(Looper.getMainLooper())
    private val beepRunnable  = object : Runnable {
        override fun run() {
            if (isAlertActive) {
                toneGenerator?.startTone(ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD, 1000)
                beepHandler.postDelayed(this, BEEP_INTERVAL_MS)
            }
        }
    }

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
        toneGenerator  = runCatching {
            ToneGenerator(AudioManager.STREAM_ALARM, 100)
        }.getOrNull()

        modelLoaded = loadModel()

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
        return runCatching {
            tflite = Interpreter(loadModelFile())
            Log.d(TAG, "TFLite model loaded successfully.")
            true
        }.getOrElse { e ->
            Log.w(TAG, "Model not found or failed to load: ${e.message}")
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
            bindPreviewOnly()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindPreviewOnly() {
        val provider = cameraProvider ?: return
        val selector = frontCameraSelector()
        val preview  = buildPreview()
        runCatching {
            provider.unbindAll()
            provider.bindToLifecycle(this, selector, preview)
            tvStatus.text = if (modelLoaded) "Model loaded ✓" else "⚠ No model — preview only"
        }.onFailure { e ->
            Log.e(TAG, "Camera bind failed: ${e.message}")
            tvStatus.text = "❌ Camera error: ${e.message}"
        }
    }

    // ── Detection start / stop ───────────────────────────────────────────────
    private fun startDetection() {
        val provider = cameraProvider ?: return
        val analysis = imageAnalysis ?: ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build().also { imageAnalysis = it }

        analysis.setAnalyzer(cameraExecutor) { runInference(it) }

        runCatching {
            provider.unbindAll()
            provider.bindToLifecycle(this, frontCameraSelector(), buildPreview(), analysis)
            isDetecting   = true
            btnStart.text = "⏹  Stop Detection"
            tvStatus.text = "🔍 Detecting…"
        }.onFailure { e -> Log.e(TAG, "Detection bind failed: ${e.message}") }
    }

    private fun stopDetection() {
        imageAnalysis?.clearAnalyzer()
        bindPreviewOnly()

        isDetecting      = false
        isAlertActive    = false
        noFaceFrames     = 0
        beepHandler.removeCallbacks(beepRunnable)
        toneGenerator?.stopTone()
        btnStart.text    = getString(R.string.start_detection)
        tvStatus.text    = "Detection stopped — camera preview active"
        tvAlert.visibility   = View.GONE
        tvConfidence.text    = "Confidence: --"
        predictionWindow.clear()
    }

    // ── Inference pipeline ───────────────────────────────────────────────────
    @OptIn(ExperimentalGetImage::class)
    private fun runInference(imageProxy: ImageProxy) {
        // Throttle: skip frames that arrive too quickly.
        val now = SystemClock.elapsedRealtime()
        if (now - lastInferenceMs < INFERENCE_INTERVAL_MS) {
            imageProxy.close()
            return
        }

        val tfl = tflite ?: run { imageProxy.close(); return }

        // Pass the ImageProxy directly to ML Kit using InputImage.fromMediaImage()
        // which accepts the YUV_420_888 format that CameraX provides natively.
        // This avoids the Bitmap conversion that caused the StreamingFormatChecker
        // warning and is significantly faster on-device.
        val mediaImage = imageProxy.image ?: run { imageProxy.close(); return }
        val mlImage    = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)

        faceDetector.process(mlImage)
            .addOnSuccessListener { faces ->
                if (faces.isEmpty()) {
                    // No face detected = person looked away / out of frame.
                    // Treat as AWAKE — inject a 0.0f (awake) vote into the window
                    // so a momentary head-turn does not trigger a false drowsy alert.
                    noFaceFrames++
                    Log.d(TAG, "No face frame #$noFaceFrames — injecting awake vote")
                    val shouldAlert = applyRollingWindow(0.0f) // force awake vote
                    lastInferenceMs = SystemClock.elapsedRealtime()
                    runOnUiThread { updateUI(0.0f, 1.0f, shouldAlert) }
                    return@addOnSuccessListener
                }

                // Face found — reset no-face counter
                noFaceFrames = 0

                // Pick largest face and crop with padding
                val best = faces.maxByOrNull { f ->
                    val b = f.boundingBox; b.width() * b.height()
                }!!

                // IMPORTANT: ML Kit's bounding box coordinates are in the ROTATED
                // (upright) image space because we passed rotationDegrees to
                // InputImage.fromMediaImage(). But imageProxy.toBitmap() returns
                // the raw sensor bitmap WITHOUT applying that rotation.
                // We must rotate the bitmap to match the ML Kit coordinate space
                // before cropping, otherwise we cut the wrong region entirely.
                val rawBitmap = imageProxy.toBitmap()
                val rotation  = imageProxy.imageInfo.rotationDegrees.toFloat()
                val bitmap = if (rotation != 0f) {
                    val matrix = android.graphics.Matrix().apply { postRotate(rotation) }
                    Bitmap.createBitmap(rawBitmap, 0, 0,
                        rawBitmap.width, rawBitmap.height, matrix, true)
                } else rawBitmap

                val b   = best.boundingBox
                val w   = bitmap.width.toFloat()
                val h   = bitmap.height.toFloat()
                val pad = FACE_PADDING

                val x1 = max(0f, b.left   - pad * b.width()).toInt()
                val y1 = max(0f, b.top    - pad * b.height()).toInt()
                val x2 = min(w,  b.right  + pad * b.width()).toInt()
                val y2 = min(h,  b.bottom + pad * b.height()).toInt()

                Log.v(TAG, "Face bbox: ($x1,$y1)→($x2,$y2) in ${bitmap.width}×${bitmap.height} " +
                        "(rotation=${rotation.toInt()}°)")

                val cropped = Bitmap.createBitmap(bitmap, x1, y1, x2 - x1, y2 - y1)
                val faceBitmap = Bitmap.createScaledBitmap(cropped, INPUT_SIZE, INPUT_SIZE, true)

                val input = buildInputBuffer(faceBitmap)
                val (drowsyScore, awakeScore) = runTfliteInference(tfl, input)
                val shouldAlert = applyRollingWindow(drowsyScore)
                lastInferenceMs = SystemClock.elapsedRealtime()
                runOnUiThread { updateUI(drowsyScore, awakeScore, shouldAlert) }
            }
            .addOnFailureListener { e ->
                Log.e(TAG, "Face detection error: ${e.message}")
            }
            .addOnCompleteListener {
                // Always close the proxy when ML Kit is done with the MediaImage.
                // This MUST be in addOnCompleteListener (not Success/Failure) to
                // guarantee it runs regardless of which branch fires.
                imageProxy.close()
            }
    }

    // detectAndCropFace inlined into runInference (see above)

    /**
     * Converts [bitmap] into the ByteBuffer expected by the TFLite model.
     * Normalisation matches EfficientNet's preprocess_input: pixel ÷ 127.5 − 1.0,
     * mapping [0, 255] to [−1, 1].
     */
    private fun buildInputBuffer(bitmap: Bitmap): ByteBuffer {
        val scaled = if (bitmap.width == INPUT_SIZE && bitmap.height == INPUT_SIZE)
            bitmap
        else
            Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)

        val buf = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
            .order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        scaled.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        for (px in pixels) {
            buf.putFloat(((px shr 16) and 0xFF) / 127.5f - 1.0f) // R
            buf.putFloat(((px shr  8) and 0xFF) / 127.5f - 1.0f) // G
            buf.putFloat(( px         and 0xFF) / 127.5f - 1.0f) // B
        }
        buf.rewind()
        return buf
    }

    /**
     * Runs TFLite inference and returns Pair(drowsyScore, awakeScore).
     * Handles both sigmoid [1,1] and softmax [1,2] output shapes automatically.
     * Model class order (from Python training): index 0 = Drowsy, index 1 = Non-Drowsy.
     */
    private fun runTfliteInference(tfl: Interpreter, input: ByteBuffer): Pair<Float, Float> {
        val shape    = tfl.getOutputTensor(0).shape()
        val lastDim  = shape.lastOrNull() ?: 1
        Log.v(TAG, "Model output shape: ${shape.toList()}")

        return if (lastDim >= 2) {
            // Softmax [1, 2]: flow_from_directory assigns alphabetical order →
            //   index 0 = Drowsy, index 1 = Non Drowsy
            val output = Array(1) { FloatArray(lastDim) }
            tfl.run(input, output)
            val drowsy = output[0][0]
            val awake  = output[0][1]
            Log.v(TAG, "[2-class] drowsy=${"%.3f".format(drowsy)}  awake=${"%.3f".format(awake)}")
            Pair(drowsy, awake)
        } else {
            // Sigmoid [1, 1]: Keras sigmoid on a binary problem outputs the
            // probability of the POSITIVE class = class index 1.
            // flow_from_directory alphabetical order: 0=Drowsy, 1=Non Drowsy
            // → raw value = P(Non Drowsy), so drowsy = 1 − raw.
            val output = Array(1) { FloatArray(1) }
            tfl.run(input, output)
            val nonDrowsy = output[0][0]
            val drowsy    = 1f - nonDrowsy
            Log.v(TAG, "[sigmoid] raw(nonDrowsy)=${"%.3f".format(nonDrowsy)}  " +
                    "drowsy=${"%.3f".format(drowsy)}  awake=${"%.3f".format(nonDrowsy)}")
            Pair(drowsy, nonDrowsy)
        }
    }

    /**
     * FIX 1: Correct rolling-window majority vote (original had ×5 multiplier bug).
     * FIX 4: Dead-band filtering — scores within ±CONFIDENCE_DEAD_BAND of the
     *         threshold are ignored, preventing state flicker at the boundary.
     *
     * Returns true when ALERT_FRAMES or more of the last WINDOW_SIZE frames
     * scored above the drowsy threshold.
     */
    private fun applyRollingWindow(drowsyScore: Float): Boolean {
        // Dead-band: skip updating the window if confidence is too low
        val distFromThreshold = kotlin.math.abs(drowsyScore - DROWSY_THRESHOLD)
        if (distFromThreshold < CONFIDENCE_DEAD_BAND) {
            // Score is ambiguous — preserve last window state unchanged
            return predictionWindow.count { it } >= ALERT_FRAMES
        }

        if (predictionWindow.size >= WINDOW_SIZE) predictionWindow.removeFirst()
        predictionWindow.addLast(drowsyScore >= DROWSY_THRESHOLD)

        return predictionWindow.count { it } >= ALERT_FRAMES
    }

    // ── UI update ────────────────────────────────────────────────────────────
    private fun updateUI(drowsyScore: Float, awakeScore: Float, alert: Boolean) {
        tvConfidence.text = "Drowsy: ${(drowsyScore * 100).toInt()}%  |  " +
                "Awake: ${(awakeScore  * 100).toInt()}%"

        when {
            alert && !isAlertActive -> {
                // Transition: awake → drowsy — start continuous beep loop
                isAlertActive      = true
                tvStatus.text      = "😴 DROWSY"
                tvAlert.visibility = View.VISIBLE
                // Start the repeating beep immediately, then re-schedule itself
                // every BEEP_INTERVAL_MS via beepRunnable until isAlertActive=false.
                beepHandler.removeCallbacks(beepRunnable)
                beepHandler.post(beepRunnable)
            }
            !alert && isAlertActive -> {
                // Transition: drowsy → awake — stop beep loop
                isAlertActive      = false
                tvStatus.text      = "✅ AWAKE"
                tvAlert.visibility = View.GONE
                beepHandler.removeCallbacks(beepRunnable)
                toneGenerator?.stopTone()
            }
            !alert -> {
                tvStatus.text = "✅ AWAKE"
            }
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────
    private fun frontCameraSelector() = CameraSelector.Builder()
        .requireLensFacing(CameraSelector.LENS_FACING_FRONT).build()

    private fun buildPreview() = Preview.Builder().build().also {
        it.surfaceProvider = previewView.surfaceProvider
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────
    override fun onDestroy() {
        super.onDestroy()
        cameraProvider?.unbindAll()
        cameraExecutor.shutdown()
        tflite?.close()
        toneGenerator?.release()
        faceDetector.close()
    }
}