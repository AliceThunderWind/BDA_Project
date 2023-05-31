package files

import org.apache.commons.lang.mutable.Mutable
import org.apache.spark.SparkContext
import org.apache.spark.sql.{ColumnName, DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.{col, regexp_replace}
import org.apache.spark.sql.functions.array_contains

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
      .getOrCreate()

    private val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("WARN")

    // Specify the path to your Parquet file
    val parquetFile = "./src/main/data/data.parquet"


    // Functions
    def main(args: Array[String]): Unit = {
        val data: DataFrame = read_file(parquetFile)
        // question 1: which year holds the highest number of produced tracks ?
        val groupedYear = groupByCount(data,"year").filter(col("year") =!= 0)
        // which country is home to the highest number of artists ?
        val groupedLocation = groupByCount(data,"artist_location")
        // which are the most popular music genre ?
        val groupedGenre = uniqueGenreCount(data, columnName = "artist_genre")

        // question 2: what is the average BPM per music genre?
        val listGenre = groupedGenre.select("term").collect().map(_.getString(0)).toList
        val avgBPM = avgMetricbyGenre(data, "tempo", listGenre)
        val avgBPMResults = printResults(avgBPM, "tempo")
        // what is the average loudness per music genre?
        val avgLoudness = avgMetricbyGenre(data, "loudness", listGenre)
        val avgLoudnessResults = printResults(avgLoudness, "loudness")

        // question 3:

        // question 4:

    }

    def read_file(string: String): DataFrame = {
        // Read the Parquet file into a DataFrame and sort it
        val data: DataFrame = Data.readParquetFile(parquetFile)
        // Get the unique values of the feature "year" and remove rows where year = 0
        return data
    }

    def groupByCount(data: DataFrame, columnName: String): DataFrame = {
        // remove rows where columnName == 0
        val filteredData: DataFrame = data.filter(col(columnName).isNotNull)
        // sort the dataframe on columnName
        val sortedData: DataFrame = filteredData.sort(col(columnName).asc)
        // group the dataframe
        val groupedData = sortedData.groupBy(columnName).count().sort(col("count").desc)
        return groupedData
    }

    def groupByMean(data: DataFrame, columnName: String): DataFrame = {
        // remove rows where columnName == 0 or Nan
        val filteredData: DataFrame = data.filter(col(columnName).isNotNull && !isnan(col(columnName)) && col(columnName) =!= 0)
        // sort the dataframe on columnName
        val sortedData: DataFrame = filteredData.sort(col(columnName).asc)
        // group the dataframe by "year"
        val groupedData = sortedData.groupBy(columnName).mean().sort(col("mean").desc)
        return groupedData
    }

    def avgMetricbyGenre(data: DataFrame, columnName: String, listGenre: List[String]): mutable.Map[String, Double] = {
        // Filter the DataFrame based on the top 10 genres and calculate the average BPM
        val top10Genres = listGenre.take(10).distinct.map(_.trim).filter(_.nonEmpty)
        val dfWithArrayGenre = data.withColumn("artist_genre_array", split(trim(col("artist_genre"), "[]"), ","))
        val result = scala.collection.mutable.Map[String, Double]() // Mutable map to store results
        for (term <- top10Genres) {
            val filteredData = dfWithArrayGenre.filter(array_contains(col("artist_genre_array"), term))
            val meanBPM = filteredData.groupBy().agg(mean(col(columnName)).as(f"mean${columnName}")).first().getAs[Double](f"mean${columnName}")
            result(term) = meanBPM
        }
        return result
    }

    def printResults(result: Map[String, Double], columnName: String): Unit = {
        result.toMap // Convert mutable map to immutable map and return
        result.foreach { case (term, mean) =>
            println(s"Music genre: $term", f" Average ${columnName}: {$mean}")
        }
    }

    def uniqueGenreCount(data: DataFrame, columnName: String): DataFrame = {
        val filteredData: DataFrame = data.filter(col(columnName).isNotNull)
        val dfExploded = filteredData.select(explode(split(col(columnName), ",")).as("term")) // Split the string into an array using "|"
        val termCounts = dfExploded.groupBy("term").agg(count("*").as("count")).orderBy(col("count").desc)
        val filteredTermCount = termCounts.withColumn("term", regexp_replace(col("term"), "[\\[\\]]", ""))
        return filteredTermCount
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
