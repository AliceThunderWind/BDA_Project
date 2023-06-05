package files

import org.apache.commons.lang.mutable.Mutable
import org.apache.spark.SparkContext
import org.apache.spark.sql.{ColumnName, DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.{col, regexp_replace}
import org.apache.spark.sql.functions.array_contains
import scala.collection.mutable.ArraySeq

import scala.collection.mutable.Map
import java.time.Instant
import scala.collection.mutable

case class PaintEvent(
                       timestamp: Instant,
                       artist_id: Types.artist_id,
                       artist_name: Types.artist_name,
                       artist_location: Types.artist_location,
                       artist_latitude: Types.artist_latitude,
                       artist_longitude: Types.artist_longitude,
                       song_id: Types.song_id,
                       title: Types.title,
                       song_hotness: Types.song_hotness,
                       similar_artists: Types.similar_artists,
                       artist_terms: Types.artist_terms,
                       artist_terms_freq: Types.artist_terms_freq,
                       artist_terms_weight: Types.artist_terms_weight,
                       duration: Types.duration,
                       time_signature : Types.time_signature,
                       time_signature_confidence : Types.time_signature_confidence,
                       beats_start : Types.beats_start,
                       beats_confidence : Types.beats_confidence,
                       key : Types.key,
                       key_confidence : Types.key_confidence,
                       loudness : Types.loudness,
                       energy : Types.energy,
                       mode : Types.mode,
                       mode_confidence : Types.mode_confidence,
                       tempo : Types.tempo,
                       year : Types.year
                     )

object Query {

    val spark: SparkSession = SparkSession.builder
      .appName("Test")
      .master("local[*]")
      .config("spark.sql.debug.maxToStringFields", "1000000")
      .getOrCreate()

    private val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("WARN")

    // Specify the path to your Parquet file
    val parquetFile = "./src/main/data/data.parquet"


    // Functions
    def main(args: Array[String]): Unit = {

    }

    def read_file(string: String): DataFrame = {
        // Read the Parquet file into a DataFrame and sort it
        val data: DataFrame = Data.readParquetFile(parquetFile)
        // Get the unique values of the feature "year" and remove rows where year = 0
        return data
    }

    val timing = new StringBuffer
    def timed[T](label: String, code: => T): T = {
        val start = System.currentTimeMillis()
        val result = code
        val stop = System.currentTimeMillis()
        timing.append(s"Processing $label took ${stop - start} ms.\n")
        result
    }
}
