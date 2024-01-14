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
import androidx.navigation.fragment.findNavController
import com.agh.federatedlearninginmcc.databinding.FragmentOcrBinding
import java.io.File
import java.io.IOException
import java.util.Date
import java.util.Locale

class OcrFragment : Fragment() {
    private lateinit var binding: FragmentOcrBinding
    private var photoURI: Uri? = null

    companion object {
        private const val TAG = "MainActivity"
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        binding = FragmentOcrBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        addOnClickedListener()
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

            gotoTrainingButton.setOnClickListener {
                findNavController().navigate(R.id.action_ocrFragment_to_trainingFragment)
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
                Toast.makeText(context, "Permission request denied", Toast.LENGTH_SHORT).show()
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
                requireContext(),
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
        ContextCompat.checkSelfPermission(requireContext(), it) == PackageManager.PERMISSION_GRANTED
    }

    @Throws(IOException::class)
    private fun createImageFile(): File {
        // Create an image file name
        val timeStamp: String = SimpleDateFormat(FILENAME_FORMAT, Locale.US).format(Date())
        val storageDir = requireContext().getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile("JPEG_${timeStamp}_", ".jpg", storageDir)
    }

}