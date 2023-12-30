package com.agh.federatedlearninginmcc.dataset

import androidx.room.*
import com.agh.federatedlearninginmcc.ml.ModelVariant
import com.agh.federatedlearninginmcc.ocr.BenchmarkInfo
import com.agh.federatedlearninginmcc.ocr.ImageInfo
import java.time.Instant
import kotlin.time.Duration
import kotlin.time.DurationUnit
import kotlin.time.toDuration

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

    override fun addBenchmarkInfo(benchmarkInfo: BenchmarkInfo) {
        db.localBenchmarkTaskInfoDao().insert(LocalBenchmarkTaskInfo(
            benchmarkInfo.repeats,
            benchmarkInfo.meanComputationTime.inWholeMilliseconds,
            Instant.now().toEpochMilli()
        ))
    }

    override fun clear() {
        db.locallyComputedTimeSampleDao().clear()
        db.cloudComputedTimeSampleDao().clear()
        db.localBenchmarkTaskInfoDao()
    }

    override fun getLocalTimeDataset(benchmarkInfo: BenchmarkInfo): Dataset {
        val samples = db.locallyComputedTimeSampleDao().getAll()
        val xs = samples.map { createLocalTimeXSample(benchmarkInfo, it.imgInfo!!) }
        val ys = samples.map { it.computationTimeMillis!! }.map { it.toFloat() }.toFloatArray()
        return Dataset(xs, ys)
    }

    override fun getCloudTimeDataset(benchmarkInfo: BenchmarkInfo): Dataset {
        val samples = db.cloudComputedTimeSampleDao().getAll()
        val xs = samples.map { createCloudTimeXSample(benchmarkInfo, it.imgInfo!!) }
        val ys = samples.map { it.computationTimeMillis!! }.map { it.toFloat() }.toFloatArray()
        return Dataset(xs, ys)
    }

    override fun getBenchmarkInfo(): BenchmarkInfo? {
        return db.localBenchmarkTaskInfoDao().getLast()?.let {
            BenchmarkInfo(it.repeats!!, it.meanComputationTimeMillis!!.toDuration(DurationUnit.MILLISECONDS))
        }
    }

    override fun getLocalTimeDatasetSize(): Int {
        return db.locallyComputedTimeSampleDao().getSize()
    }

    override fun getCloudTimeDatasetSize(): Int {
        return db.cloudComputedTimeSampleDao().getSize()
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

@Entity
data class LocalBenchmarkTaskInfo(
    @PrimaryKey(autoGenerate = true) val id: Int,
    val repeats: Int?,
    val meanComputationTimeMillis: Long?,
    val utcTimestamp: Long?
) {
    constructor(repeats: Int?, meanComputationTimeMillis: Long?, utcTimestamp: Long?):
            this(0, repeats, meanComputationTimeMillis, utcTimestamp)
}

@Dao
interface LocallyComputedTimeSampleDao {
    @Insert
    fun insert(sample: LocallyComputedTimeSample)

    @Query("SELECT * FROM LocallyComputedTimeSample")
    fun getAll(): List<LocallyComputedTimeSample>

    @Query("DELETE FROM LocallyComputedTimeSample WHERE 1=1")
    fun clear()

    @Query("SELECT COUNT(*) FROM LocallyComputedTimeSample")
    fun getSize(): Int
}

@Dao
interface CloudComputedTimeSampleDao {
    @Insert
    fun insert(sample: CloudComputedTimeSample)

    @Query("SELECT * FROM CloudComputedTimeSample")
    fun getAll(): List<CloudComputedTimeSample>

    @Query("DELETE FROM CloudComputedTimeSample WHERE 1=1")
    fun clear()

    @Query("SELECT COUNT(*) FROM CloudComputedTimeSample")
    fun getSize(): Int
}

@Dao
interface LocalBenchmarkTaskInfoDao {
    @Insert
    fun insert(sample: LocalBenchmarkTaskInfo)

    @Query("SELECT * FROM LocalBenchmarkTaskInfo ORDER BY utcTimestamp DESC LIMIT 1")
    fun getLast(): LocalBenchmarkTaskInfo?

    @Query("DELETE FROM LocalBenchmarkTaskInfo WHERE 1=1")
    fun clear()
}

@Database(entities = arrayOf(
    LocallyComputedTimeSample::class,
    CloudComputedTimeSample::class,
    LocalBenchmarkTaskInfo::class,
), version = 2)
abstract class OcrDatabase: RoomDatabase() {
    abstract fun locallyComputedTimeSampleDao(): LocallyComputedTimeSampleDao
    abstract fun cloudComputedTimeSampleDao(): CloudComputedTimeSampleDao
    abstract fun localBenchmarkTaskInfoDao(): LocalBenchmarkTaskInfoDao
}
