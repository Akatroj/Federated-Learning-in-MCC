package com.agh.federatedlearninginmcc.dataset

import androidx.room.*
import com.agh.federatedlearninginmcc.ml.ModelVariant
import com.agh.federatedlearninginmcc.ocr.ImageInfo
import kotlin.time.Duration

// not the most efficient way to store it but whatever

class SqlOcrDataset(private val db: OcrDatabase): OCRDataset() {
    override fun addLocallyComputedTimeSample(imgInfo: ImageInfo, computationTime: Duration) {
        val sample = LocallyComputedTimeSample(imgInfo, computationTime.inWholeMilliseconds.toInt())
        db.locallyComputedTimeSampleDao().insert(sample)
        notifyDatasetUpdated(ModelVariant.LOCAL_TIME)
    }

    override fun addCloudComputedTimeSample(
        imgInfo: ImageInfo,
        totalComputationTime: Duration
    ) {
        val sample = CloudComputedTimeSample(imgInfo, totalComputationTime.inWholeMilliseconds.toInt())
        db.cloudComputedTimeSampleDao().insert(sample)
        notifyDatasetUpdated(ModelVariant.CLOUD_TIME)
    }

    override fun clear() {
        db.locallyComputedTimeSampleDao().clear()
        db.cloudComputedTimeSampleDao().clear()
    }

    override fun getLocalTimeDataset(): Pair<List<FloatArray>, FloatArray> {
        val samples = db.locallyComputedTimeSampleDao().getAll()
        val xs = samples.map { convertLocallyComputedTimeSample(it.imgInfo!!) }
        val ys = samples.map { it.computationTimeMillis!! }.map { it.toFloat() }.toFloatArray()
        return Pair(xs, ys)
    }

    override fun getCloudTimeDataset(): Pair<List<FloatArray>, FloatArray> {
        val samples = db.cloudComputedTimeSampleDao().getAll()
        val xs = samples.map { convertLocallyComputedTimeSample(it.imgInfo!!) }
        val ys = samples.map { it.computationTimeMillis!! }.map { it.toFloat() }.toFloatArray()
        return Pair(xs, ys)
    }
}

@Entity
data class LocallyComputedTimeSample(
    @PrimaryKey(autoGenerate = true) val sampleId: Int,
    @Embedded val imgInfo: ImageInfo?,
    val computationTimeMillis: Int?
) {
    constructor(imgInfo: ImageInfo?, computationTimeMillis: Int?):
            this(0, imgInfo, computationTimeMillis)
}

@Entity
data class CloudComputedTimeSample(
    @PrimaryKey(autoGenerate = true) val sampleId: Int,
    @Embedded val imgInfo: ImageInfo?,
    val computationTimeMillis: Int?
    // TODO info related to cloud
) {
    constructor(imgInfo: ImageInfo?, computationTimeMillis: Int?):
            this(0, imgInfo, computationTimeMillis)
}

@Dao
interface LocallyComputedTimeSampleDao {
    @Insert
    fun insert(sample: LocallyComputedTimeSample)

    @Query("SELECT * FROM LocallyComputedTimeSample")
    fun getAll(): List<LocallyComputedTimeSample>

    @Query("DELETE FROM LocallyComputedTimeSample WHERE 1=1")
    fun clear()
}

@Dao
interface CloudComputedTimeSampleDao {
    @Insert
    fun insert(sample: CloudComputedTimeSample)

    @Query("SELECT * FROM CloudComputedTimeSample")
    fun getAll(): List<CloudComputedTimeSample>

    @Query("DELETE FROM CloudComputedTimeSample WHERE 1=1")
    fun clear()
}

@Database(entities = arrayOf(LocallyComputedTimeSample::class, CloudComputedTimeSample::class), version = 1)
abstract class OcrDatabase: RoomDatabase() {
    abstract fun locallyComputedTimeSampleDao(): LocallyComputedTimeSampleDao
    abstract fun cloudComputedTimeSampleDao(): CloudComputedTimeSampleDao
}
