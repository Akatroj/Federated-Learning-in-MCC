package com.agh.federatedlearninginmcc

import android.os.BatteryManager
import android.os.Bundle
import android.util.Log
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.navigation.fragment.findNavController
import androidx.room.Room
import com.agh.federatedlearninginmcc.databinding.FragmentTrainingBinding
import com.agh.federatedlearninginmcc.dataset.OCRDataset
import com.agh.federatedlearninginmcc.dataset.OcrDatabase
import com.agh.federatedlearninginmcc.dataset.SqlOcrDataset
import com.agh.federatedlearninginmcc.ml.InferenceEngine
import com.agh.federatedlearninginmcc.ml.ModelFactory
import com.agh.federatedlearninginmcc.ml.ModelVariant
import com.agh.federatedlearninginmcc.ml.RunningLocation
import com.agh.federatedlearninginmcc.ml.TrainingEngine
import com.agh.federatedlearninginmcc.ocr.BenchmarkHandler
import com.agh.federatedlearninginmcc.ocr.K8sOCREngine
import com.agh.federatedlearninginmcc.ocr.LocalOCREngine
import com.agh.federatedlearninginmcc.ocr.OCRService
import com.agh.federatedlearninginmcc.ocr.TesseractOCREngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File

class TrainingFragment : Fragment() {
    private lateinit var binding: FragmentTrainingBinding
    private lateinit var db: OcrDatabase
    private lateinit var dataset: OCRDataset
    private lateinit var localOcr: LocalOCREngine
    private lateinit var benchmarkHandler: BenchmarkHandler

    companion object {
        private const val K8S_OCR_URL = "http://172.18.0.3:31555/base64"
        private const val FLOWER_SERVER_IP = "10.0.2.2" // localhost from emulator
        private const val TOTAL_IMAGES = 300
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        binding = FragmentTrainingBinding.inflate(inflater, container, false)
        initTrainingRequirements()
        return binding.root
    }

    private fun initTrainingRequirements() {
        db = Room.databaseBuilder(requireContext(), OcrDatabase::class.java, "ocrdatabase")
            .allowMainThreadQueries()
            .fallbackToDestructiveMigration()
            .build()
        dataset = SqlOcrDataset(db)
        localOcr = initTesseractOcr()
        benchmarkHandler = BenchmarkHandler(dataset, localOcr, getBenchmarkImage())
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        updateDatasetStatus()
        binding.apply {
            gotoOcrButton.setOnClickListener {
                findNavController().navigate(R.id.action_trainingFragment_to_ocrFragment)
            }
            prepareDatasetBtn.setOnClickListener {
                val deviceId = deviceIdInput.text.toString().toInt()
                val totalDevices = totalDevicesInput.text.toString().toInt()
                assert(deviceId in 0..<totalDevices)
                prepareTrainingDataset(dataset, deviceId, totalDevices, TOTAL_IMAGES)
            }
            joinTrainingBtn.setOnClickListener {
                joinTrainingBtn.isEnabled = false
                joinTraining()
            }
            testEnergyBtn.setOnClickListener {
                testEnergyBtn.isEnabled = false
                testEnergy()
            }
            testTrainedOnStoredImages.setOnClickListener {
                testModels(getImagesForTesting(), trained = true)
            }
            testUntrainedOnStoredImages.setOnClickListener {
                testModels(getImagesForTesting(), trained = false)
            }
            clearDatasetBtn.setOnClickListener {
                lifecycleScope.launch(Dispatchers.IO) {
                    dataset.clear()
                    launch(Dispatchers.Main) {
                        updateDatasetStatus()
                    }
                }
            }
        }
    }

    private fun initTesseractOcr(): TesseractOCREngine {
        val tessDataDir = File(requireContext().filesDir, "tessdata")
        if (!tessDataDir.exists()) {
            val tesseractTrainData = File(tessDataDir, "eng.traineddata")
            tessDataDir.mkdir()
            requireContext().assets.open(tesseractTrainData.name).use { inputStream ->
                tesseractTrainData.outputStream().use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
        }
        return TesseractOCREngine(requireContext().filesDir.path)
    }

    private fun updateDatasetStatus() {
        val localTimeSize = dataset.getDatasetSize(ModelVariant.LOCAL_TIME)
        val cloudTimeSize = dataset.getDatasetSize(ModelVariant.CLOUD_COMPUTATION_TIME)

        binding.datasetStatus.text = "Dataset status: local time samples = $localTimeSize, cloud time samples = $cloudTimeSize"
    }

    private fun getImagesForTesting() = (1100..<1140).map { getOcrImageFile(it) }

    private fun getBenchmarkImage(): File {
        return getOcrImageFile(0)
    }

    private fun getOcrImageFile(imgId: Int): File {
        val imgName = "img_$imgId.jpg"
        val assetPath = "preprocessed/$imgName"
        val testImgFile = File(requireContext().filesDir, imgName)
        requireContext().assets.open(assetPath).use { inputStream ->
            testImgFile.outputStream().use { outputStream ->
                inputStream.copyTo(outputStream)
            }
        }
        return testImgFile
    }

    private fun createOCRService(dataset: OCRDataset, forcedLocation: RunningLocation = RunningLocation.PREDICT, restoreModels: Boolean = true): OCRService {
        return OCRService(
            initTesseractOcr(),
            K8sOCREngine(K8S_OCR_URL),
            InferenceEngine(
                dataset,
                ModelFactory.createModel(
                    requireContext(),
                    ModelVariant.LOCAL_TIME,
                    restoreModels
                ),
                ModelFactory.createModel(
                    requireContext(),
                    ModelVariant.CLOUD_COMPUTATION_TIME,
                    restoreModels
                ),
                ModelFactory.createModel(
                    requireContext(),
                    ModelVariant.CLOUD_TRANSMISSION_TIME,
                    restoreModels
                ),
                forcedLocation
            ),
            dataset,
        )
    }

    private fun testModels(imgs: List<File> = listOf(getBenchmarkImage()), trained: Boolean = true) {
        lifecycleScope.launch(Dispatchers.IO) {
            benchmarkHandler.assertHasRunBenchmark()
            val ocrService = createOCRService(dataset, RunningLocation.PREDICT, trained)

            imgs.forEach {
                Log.d("OCR", "Launching OCR")
                val res = ocrService.doOCR(it)
                Log.d("OCR", res)
            }

            val summary = ocrService.printStatsAndGetSummary()
            launch(Dispatchers.Main) {
                Toast.makeText(
                    context,
                    "Testing result: $summary",
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }

    private fun prepareTrainingDataset(dataset: OCRDataset, deviceId: Int, numDevices: Int = 2, totalImages: Int = TOTAL_IMAGES) {
        val firstImg = (totalImages / numDevices) * deviceId
        val lastImg = ((totalImages / numDevices) * (deviceId + 1)).coerceAtMost(totalImages)
        Log.d("DATASET_PREP", "DeviceId=$deviceId taking split ($firstImg, $lastImg) from $totalImages")

        val imgFiles = (firstImg..<lastImg).map {
            val testImgFile = File(requireContext().filesDir, "img_$it.jpg")
            requireContext().assets.open("preprocessed/img_$it.jpg").use { inputStream ->
                testImgFile.outputStream().use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
            testImgFile
        }.toList()

        Toast.makeText(
            context,
            "Preparing datasets, this will take a while, check logs...",
            Toast.LENGTH_LONG
        ).show()

        lifecycleScope.launch(Dispatchers.IO) {
            listOf(
                RunningLocation.FORCE_LOCAL,
                RunningLocation.FORCE_CLOUD
            ).forEach { forcedLocation ->
                val ocrService = createOCRService(dataset, forcedLocation)
                imgFiles.forEach { ocrService.doOCR(it) }
            }

            launch(Dispatchers.Main) {
                Toast.makeText(
                    context,
                    "dataset ready",
                    Toast.LENGTH_LONG
                ).show()
                updateDatasetStatus()
            }
            Log.d("OCR", "dataset ready")
        }
    }

    private fun joinTraining() {
        Toast.makeText(
            context,
            "Joining training, check logs...",
            Toast.LENGTH_LONG
        ).show()

        lifecycleScope.launch(Dispatchers.IO) {
            Log.d("OCR", "Joining federated training")
            benchmarkHandler.assertHasRunBenchmark()
            val trainer = TrainingEngine(requireContext(), FLOWER_SERVER_IP, dataset,
                minSamplesToJoinTraining = 100, restoreTrainedModel = false)
            trainer.joinFederatedTraining()
        }
    }

    private fun testEnergy() {
        // on my phone it requires ~60 OCRs and about 3 minutes to drain a single battery %
        // so imo it's totally pointless to try building energy models
        Toast.makeText(context, "Starting testing energy, running OCRs, check logs...", Toast.LENGTH_SHORT).show()

        val mBatteryManager = requireContext().getSystemService(AppCompatActivity.BATTERY_SERVICE) as BatteryManager
        val initialBatteryLevel = mBatteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        lifecycleScope.launch(Dispatchers.IO) {
            var i = 0
            val start = System.currentTimeMillis()
            while (true) {
                i++
                if (i % 3 == 0) {
                    Log.d("BAT_OCR", i.toString())
                }
                localOcr.doOCR(getBenchmarkImage())
                val batteryLevel = mBatteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
                if (batteryLevel != initialBatteryLevel) {
                    break
                }
            }
            val totalTime = System.currentTimeMillis() - start
            Log.d("BAT_OCR", "Drained 1% battery in $totalTime, required $i OCRs")

            launch(Dispatchers.Main) {
                Toast.makeText(
                    context,
                    "Drained 1% battery in $totalTime, required $i OCRs",
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }
}
