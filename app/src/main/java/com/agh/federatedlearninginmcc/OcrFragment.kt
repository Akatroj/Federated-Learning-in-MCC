package com.agh.federatedlearninginmcc

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.icu.text.SimpleDateFormat
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.net.toFile
import androidx.lifecycle.lifecycleScope
import androidx.navigation.fragment.findNavController
import com.agh.federatedlearninginmcc.databinding.FragmentOcrBinding
import com.agh.federatedlearninginmcc.dataset.OcrDatabase
import com.agh.federatedlearninginmcc.dataset.SqlOcrDataset
import com.agh.federatedlearninginmcc.ml.RunningLocation
import com.agh.federatedlearninginmcc.ocr.OCRService
import com.agh.federatedlearninginmcc.ocr.OCRServiceFactory
import com.agh.federatedlearninginmcc.ocr.TransmissionTester
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File
import java.io.IOException
import java.util.Date
import java.util.Locale

class OcrFragment : Fragment() {
    private lateinit var ocrService: OCRService
    private lateinit var binding: FragmentOcrBinding

    private var photoURI: Uri? = null

    companion object {
        private const val TAG = "OCRFragment"
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View? {
        binding = FragmentOcrBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        addOnClickedListener()
        setupOCR()
    }

    private fun setupOCR() {
        lifecycleScope.launch(Dispatchers.IO) {
            val db = OcrDatabase.getInstance(requireContext())
            val dataset = SqlOcrDataset(db)
            ocrService = OCRServiceFactory.create(requireContext(), dataset)
        }
    }

    private fun addOnClickedListener() {
        binding.apply {
            takePicture.setOnClickListener {
                if (!allPermissionsGranted()) requestPermissions()
                dispatchTakePictureIntent()
            }

            uploadFromGallery.setOnClickListener {
                getContentActivityLauncher.launch("image/*")
            }

            performOcrButton.setOnClickListener {
                if (photoURI == null) {
                    Toast.makeText(context, "No image selected", Toast.LENGTH_SHORT).show()
                    return@setOnClickListener
                }
                val result = ocrService.doOCR(uriToFile(photoURI!!))

                val runningLocation = if (result.prediction.shouldRunLocally) "LOCAL" else "CLOUD"

                resultText.visibility = View.VISIBLE
                infoText.visibility = View.VISIBLE

                infoText.text = """
                    Predicted local time ${result.prediction.localTime}ms
                    Predicted cloud time ${result.prediction.cloudTime}ms
                    Picked $runningLocation execution
                    OCR time ${result.actualTimeMs}ms
                """.trimIndent()
                resultText.text = "OCR result: ${result.result}"

            }

            gotoTrainingButton.setOnClickListener {
                findNavController().navigate(R.id.action_ocrFragment_to_trainingFragment)
            }
        }
    }

    private val getContentActivityLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
            photoURI = uri
            setPreview()
        }

    private val imageCaptureActivityLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                setPreview()
            }
        }

    private val permissionRequestActivityLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        // Handle Permission granted/rejected
        val permissionsGranted = permissions.entries.all { it.value }

        if (!permissionsGranted) {
            Toast.makeText(context, "Permission request denied", Toast.LENGTH_SHORT).show()
        }
    }

    private fun dispatchTakePictureIntent() {
        val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        Log.i(TAG, "takePictureIntent: $takePictureIntent")
        // Create the File where the photo should go
        val photoFile: File = try {
            createImageFile()
        } catch (ex: IOException) {
            Toast.makeText(context, "Error occurred while creating the File", Toast.LENGTH_SHORT)
                .show()
            return
        }


        photoURI = FileProvider.getUriForFile(
            requireContext(), "com.agh.federatedlearninginmcc.fileprovider", photoFile
        )
        Log.i(TAG, "photoURI: $photoURI")
        takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
        imageCaptureActivityLauncher.launch(takePictureIntent)
    }


    private fun requestPermissions() =
        permissionRequestActivityLauncher.launch(REQUIRED_PERMISSIONS)

    private fun allPermissionsGranted(): Boolean = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(requireContext(), it) == PackageManager.PERMISSION_GRANTED
    }

    private fun setPreview() {
        if (photoURI == null) return
        binding.apply {
            imageView.setImageURI(photoURI)
            previewText.visibility = View.VISIBLE
            imageView.visibility = View.VISIBLE
            performOcrButton.visibility = View.VISIBLE
        }
    }

    private fun uriToFile(uri: Uri): File {
        val stream = requireContext().contentResolver.openInputStream(uri)
        val file = createImageFile()
        file.outputStream().use { stream?.copyTo(it) }
        return file
    }

    @Throws(IOException::class)
    private fun createImageFile(): File {
        // Create an image file name
        val timeStamp: String = SimpleDateFormat(FILENAME_FORMAT, Locale.US).format(Date())
        val storageDir = requireContext().getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile("JPEG_${timeStamp}_", ".jpg", storageDir)
    }

}