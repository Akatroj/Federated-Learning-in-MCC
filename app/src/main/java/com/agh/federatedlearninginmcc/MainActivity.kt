package com.agh.federatedlearninginmcc

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.icu.text.SimpleDateFormat
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import androidx.room.Room
import com.agh.federatedlearninginmcc.databinding.ActivityMainBinding
import com.agh.federatedlearninginmcc.dataset.OCRDataset
import com.agh.federatedlearninginmcc.dataset.OcrDatabase
import com.agh.federatedlearninginmcc.dataset.SqlOcrDataset
import com.agh.federatedlearninginmcc.ml.*
import com.agh.federatedlearninginmcc.ocr.K8sOCREngine
import com.agh.federatedlearninginmcc.ocr.OCRService
import com.agh.federatedlearninginmcc.ocr.TesseractOCREngine
import com.example.tfltest.FmnistFederatedTesting
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.File
import java.io.IOException
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private var photoURI: Uri? = null

    companion object {
        private const val TAG = "MainActivity"
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)

        setContentView(binding.root)
        addOnClickedListener()
        testNonUIStuff()
    }

    private fun addOnClickedListener() {
        binding.apply {
            takePicture.setOnClickListener {
                if (allPermissionsGranted()) {
                    dispatchTakePictureIntent()
                } else {
                    requestPermissions()
                }
            }
            uploadFromGallery.setOnClickListener {
                getContentActivityLauncher.launch("image/*")
            }
        }
    }

    private val getContentActivityLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
            binding.imageView.setImageURI(uri)
        }

    private val imageCaptureActivityLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
//                val data: Intent? = result.data
//                val thumbnail = data?.extras?.get("data") as Bitmap
                binding.imageView.setImageURI(photoURI)
            }
        }

    private val permissionRequestActivityLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        )
        { permissions ->
            // Handle Permission granted/rejected
            val permissionsGranted = permissions.entries.all { it.value }

            if (!permissionsGranted) {
                Toast.makeText(baseContext, "Permission request denied", Toast.LENGTH_SHORT).show()
            }
        }

    private fun dispatchTakePictureIntent() {
        val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        Log.i(TAG, "takePictureIntent: $takePictureIntent")
        // Create the File where the photo should go
        val photoFile: File? = try {
            createImageFile()
        } catch (ex: IOException) {
            null
        }
        // Continue only if the File was successfully created
        photoFile?.also {
            photoURI = FileProvider.getUriForFile(
                this,
                "com.agh.federatedlearninginmcc.fileprovider",
                it
            )
            Log.i(TAG, "photoURI: $photoURI")
            takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
            imageCaptureActivityLauncher.launch(takePictureIntent)
        }
    }

    private fun requestPermissions() =
        permissionRequestActivityLauncher.launch(REQUIRED_PERMISSIONS)

    private fun allPermissionsGranted(): Boolean = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    @Throws(IOException::class)
    private fun createImageFile(): File {
        // Create an image file name
        val timeStamp: String = SimpleDateFormat(FILENAME_FORMAT, Locale.US).format(Date())
        val storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile("JPEG_${timeStamp}_", ".jpg", storageDir)
    }

    private fun testNonUIStuff() {
        // filter logcat with tag:FmnistTesting
//         testFlowerOnFmnist()
//         testTensorflowPredictionOnFmnist()
//         testTensorflowLocalTrainingOnFmnist()

//        testOCR()
//        testOCR2()
        testOcrFlowerTraining()
    }

    private fun testFlowerOnFmnist() {
        // run python server locally before running this
        val tester = FmnistFederatedTesting(this, "10.0.2.2", 8085)
        lifecycleScope.launch(Dispatchers.IO) {
            // depending on the server configuration the line below suffices to start the federated training
            // server should print some training logs and exit if everything went properly
            tester.connect()
        }
    }

    private fun testTensorflowPredictionOnFmnist(useLocallyTrainedModel: Boolean = false) {
        val tester = FmnistFederatedTesting(this, "10.0.2.2", 8085)
        lifecycleScope.launch {
            tester.apply {
                val interp = loadModel(useLocallyTrainedModel)
                testTensorflowPrediction(interp, 2)
            }
        }
    }

    private fun testTensorflowLocalTrainingOnFmnist() {
        val tester = FmnistFederatedTesting(this, "10.0.2.2", 8085)
        lifecycleScope.launch {
            tester.apply {
                var interp = loadModel(false)
                testTensorflowLearning(interp)
                writeModel(interp)
                interp = loadModel(true)
                testTensorflowLearning(interp) // should have lower loss since it was trained for a while
            }
        }
    }

    private fun initTesseractOcr(): TesseractOCREngine {
        val tessDataDir = File(filesDir, "tessdata")
        if (!tessDataDir.exists()) {
            val tesseractTrainData = File(tessDataDir, "eng.traineddata")
            tessDataDir.mkdir()
            assets.open(tesseractTrainData.name).use { inputStream ->
                tesseractTrainData.outputStream().use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
        }
        return TesseractOCREngine(filesDir.path)
    }

    private fun getTestOcrImage(): File {
        val testImgFile = File(filesDir, "ocr_test_img.png")
        if (!testImgFile.exists()) {
            assets.open(testImgFile.name).use { inputStream ->
                testImgFile.outputStream().use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
        }
        return testImgFile
    }

    private fun testOCR() {
        lifecycleScope.launch(Dispatchers.IO){
            Log.d("OCR", "Launching OCR")
            val res = initTesseractOcr().doOCR(getTestOcrImage())
            Log.d("OCR", res)
        }
    }

    private fun testOCR2() {
        val db = Room.databaseBuilder(applicationContext, OcrDatabase::class.java, "ocrdatabase")
            .allowMainThreadQueries()
            .fallbackToDestructiveMigration()
            .build()
        val dataset = SqlOcrDataset(db)

        val placeholderStats: (Int) -> NormalizationStats = {
            NormalizationStats((0..<it).map { .0f }.toFloatArray(), (0..<it).map { 1.0f }.toFloatArray(), .0f, 1.0f)
        }

        val ocrService = OCRService(
            initTesseractOcr(),
            K8sOCREngine("http://172.18.0.3:31555/base64"),
            InferenceEngine(
                dataset,
                ModelFactory.createModel(
                    applicationContext,
                    ModelVariant.LOCAL_TIME,
                    placeholderStats(ModelVariant.LOCAL_TIME.modelConfig.inputDimensions)
                ),
                ModelFactory.createModel(
                    applicationContext,
                    ModelVariant.CLOUD_TIME,
                    placeholderStats(ModelVariant.CLOUD_TIME.modelConfig.inputDimensions)
                )
            ),
            dataset
        )
        lifecycleScope.launch(Dispatchers.IO) {
            Log.d("OCR", "Launching OCR")
            val res = ocrService.doOCR(getTestOcrImage())
            Log.d("OCR", res)
        }
    }

    private fun prepareTestingDataset(dataset: OCRDataset, repeats: Int = 10) {
        dataset.clear()
        val imgFiles = (1..<11) .map {
            val testImgFile = File(filesDir, "ocr_test_img_$it.png")
            assets.open("ocr_test_images/t$it.png").use { inputStream ->
                testImgFile.outputStream().use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
            testImgFile
        }.toList()

         val placeholderStats: (Int) -> NormalizationStats = {
            NormalizationStats((0..<it).map { .0f }.toFloatArray(), (0..<it).map { 1.0f }.toFloatArray(), .0f, 1.0f)
        }

        listOf(RunningLocation.FORCE_LOCAL, RunningLocation.FORCE_CLOUD).forEach { forcedLocation ->
            val ocrService = OCRService(
                initTesseractOcr(),
                K8sOCREngine("http://172.18.0.3:31555/base64"),
                InferenceEngine(
                    dataset,
                    ModelFactory.createModel(
                        applicationContext,
                        ModelVariant.LOCAL_TIME,
                        placeholderStats(ModelVariant.LOCAL_TIME.modelConfig.inputDimensions)
                    ),
                    ModelFactory.createModel(
                        applicationContext,
                        ModelVariant.CLOUD_TIME,
                        placeholderStats(ModelVariant.CLOUD_TIME.modelConfig.inputDimensions)
                    ),
                    forcedLocation
                ),
                dataset,
            )

            for (i in 0..<repeats) {
                imgFiles.forEach { ocrService.doOCR(it) }
            }
        }
        Log.d("OCR", "dataset ready")
    }

    private fun testOcrFlowerTraining() {
        val db = Room.databaseBuilder(applicationContext, OcrDatabase::class.java, "ocrdatabase")
            .allowMainThreadQueries()
            .fallbackToDestructiveMigration()
            .build()
        val dataset = SqlOcrDataset(db)
        if (dataset.getCloudTimeDataset().xs.size < 100) {
            dataset.clear()
            lifecycleScope.launch(Dispatchers.IO) {
                Log.d("OCR", "preparing dataset")
                prepareTestingDataset(dataset)
            }
        }

        val trainer = TrainingEngine(applicationContext, "10.0.2.2", dataset, minSamplesToJoinTraining = 10)
        lifecycleScope.launch(Dispatchers.IO) {
            Log.d("OCR", "Joining federated training")
            trainer.joinFederatedTraining()
        }
    }
}
