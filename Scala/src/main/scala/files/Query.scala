package files

import org.apache.commons.lang.mutable.Mutable
import org.apache.spark.SparkContext
import org.apache.spark.sql.{ColumnName, DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.{col, regexp_replace}
import org.apache.spark.sql.functions.array_contains
import scala.collection.mutable.ArraySeq
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.feature.MinMaxScalerModel
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.linalg.Vector
import scala.collection.mutable.ArrayBuffer

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
        val data: DataFrame = read_file(parquetFile)
        // question 1: which year holds the highest number of produced tracks ?
        val groupedYear = groupByCount(data,"year").filter(col("year") =!= 0)
        // which country is home to the highest number of artists ?
        val groupedLocation = groupByCount(data,"artist_location")
        // which are the most popular music genre ?
        val groupedGenre = uniqueGenreCount(data, columnName = "artist_terms")
        println(groupedGenre.count)
        groupedGenre.show(10)

        // question 2: what is the average BPM per music genre?
        val listGenre = groupedGenre.select("term").collect().map(_.getString(0)).toList
        val avgBPM = avgMetricbyGenre(data, "tempo", listGenre)
        val avgBPMResults: Unit = printResults(avgBPM, "tempo")
        // what is the average loudness per music genre?
        val avgLoudness = avgMetricbyGenre(data, "loudness", listGenre)
        val avgLoudnessResults: Unit = printResults(avgLoudness, "loudness")

        // question 3: how to predict the music genre from features like dB, tempo, scale etc.
        // Select meaningful columns for the clustering
        val features_for_kmeans = Array("duration", "key", "loudness", "tempo", "time_signature")
        val scaledData = preprocess_data(data, features_for_kmeans)
        val dataset = data.select(features_for_kmeans.head, features_for_kmeans.tail: _*)

        // Find the best silhouettes
        val silhouettes = get_silhouettes(scaledData, 2, 10)
        println(s"Silhouettes with squared euclidean distance :")
        println(silhouettes.mkString(", "))

        // Trains a k-means model.
        val nClusters = 6
        val predictions = kmeans_predict_show(scaledData, nClusters, features_for_kmeans)

        // Print the song of a playlist
        val features_to_show = Array("artist_name", "title", "duration", "tempo", "artist_genre")
        show_predicted_musics(data, predictions, 0, 10, features_to_show)
        show_predicted_musics(data, predictions, 5, 10, features_to_show)
    }

    private def preprocess_data(data: DataFrame, columns: Array[String]): DataFrame = {
        // Select meaningful columns for the clustering
        val my_features = Array("duration", "key", "loudness", "tempo", "time_signature")
        val dataset = data.select(my_features.head, my_features.tail: _*)

        // Define the assembler
        val assembler = new VectorAssembler()
          .setInputCols(my_features)
          .setOutputCol("features")

        // Transform the DataFrame using the VectorAssembler
        val assembledData = assembler.transform(dataset)

        // Create a StandardScaler instance
        val scaler = new StandardScaler()
          .setInputCol("features")
          .setOutputCol("scaledFeatures")
          .setWithMean(true) // Optionally remove the mean from the feature vector
          .setWithStd(true) // Optionally scale the features to unit standard deviation

        // Compute summary statistics and generate the scaler model
        val scalerModel = scaler.fit(assembledData)

        // Transform the DataFrame to apply scaling
        val scaledData: DataFrame = scalerModel.transform(assembledData)
        return scaledData
    }
    private def kmeans_prediction(data: DataFrame, nClusters: Int): DataFrame = {
        // Trains a k-means model.
        val kmeans = new KMeans().setK(nClusters)
          .setFeaturesCol("scaledFeatures")
          .setSeed(1L)
        val model = kmeans.fit(data)

        // Make predictions
        val predictions = model.transform(data)
        return predictions
    }

    private def kmeans_predict_show(data: DataFrame, nClusters: Int, features_for_kmeans: Array[String]): DataFrame = {
        // Trains a k-means model.
        val kmeans = new KMeans().setK(nClusters)
          .setFeaturesCol("scaledFeatures")
          .setSeed(1L)
        val model = kmeans.fit(data)

        // Make predictions
        val predictions = model.transform(data)

        // Shows the result.
        println("Cluster Centers: ")
        println(features_for_kmeans.mkString(", "))
        model.clusterCenters.foreach(println)
        return predictions
    }

    private def get_silhouettes(data: DataFrame, minClusters: Int, maxClusters: Int): ArrayBuffer[Double]  = {
        // Loop and store results
        val silhouettes = ArrayBuffer[Double]() // Mutable collection to store results

        for (nClusters <- minClusters to maxClusters) {
            val predictions = kmeans_prediction(data, nClusters) // Call the function with the current value
            // Evaluate clustering by computing Silhouette score
            val evaluator = new ClusteringEvaluator()

            val silhouette = evaluator.evaluate(predictions)
            silhouettes += silhouette // Store the result in the collection
        }
        return silhouettes
    }

    private def show_predicted_musics(data: DataFrame, predictions: DataFrame, cluster_id: Int, musics_to_show: Int, features_to_print: Array[String]) = {
        // Add row index to data
        val dataWithIndex = data.withColumn("index", monotonically_increasing_id())
        // Add row index to predictions
        val predictionCol = predictions.select("prediction")
        val predictionsWithIndex = predictionCol.withColumn("index", monotonically_increasing_id())
        val filteredSongsDF = dataWithIndex.join(predictionsWithIndex, Seq("index"))
          .filter(col("prediction") === cluster_id)
        val filteredSongsFeatures = filteredSongsDF.select(features_to_print.head, features_to_print.tail: _*)
        println(s"First $musics_to_show musics of cluster $cluster_id :")
        filteredSongsFeatures.show(musics_to_show, false)
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

    def uniqueGenreCount(data: DataFrame, columnName: String): DataFrame = {
        val filteredData: DataFrame = data.filter(col(columnName).isNotNull)
        val dfExploded = filteredData.select(explode(split(col(columnName), ",")).as("term")) // Split the string into an array using "|"
        val termCounts = dfExploded.groupBy("term").agg(count("*").as("count")).orderBy(col("count").desc)
        val filteredTermCount = termCounts.withColumn("term", regexp_replace(col("term"), "[\\[\\]]", ""))
        return filteredTermCount
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
        val top10Genres = listGenre.take(10).distinct.map(_.trim).filter(_.nonEmpty).map(_.trim.toLowerCase)
        val dfWithArrayGenre: DataFrame = data.withColumn("artist_terms_array", split(trim(col("artist_terms"), "[]"), ","))
        val result = scala.collection.mutable.Map[String, Double]() // Mutable map to store results
        // Filter the DataFrame based on the top 10 genres and calculate the average metric
        for (term <- top10Genres) {
            var count = 0
            var metricTotal = 0.0
            val rows = dfWithArrayGenre.collect()
            rows.foreach { row =>
                val terms = row.getAs[mutable.WrappedArray[String]]("artist_terms_array").map(_.trim.toLowerCase)
                val metric = row.getAs[Double](columnName)
                if (terms.contains(term)) {
                    count += 1 // Increment count if the row contains the term
                    metricTotal += metric
                }
            }
            val avgMetric: Double = if (count > 0) metricTotal / count else 0.0 // Calculate the average metric by dividing the total by the count
            result(term) = avgMetric
        }
        return result
    }

    def printResults(result: Map[String, Double], columnName: String): Unit = {
        result.toMap // Convert mutable map to immutable map and return
        result.foreach { case (term, mean) =>
            println(s"Music genre: $term", f" Average ${columnName}: {$mean}")
        }
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
