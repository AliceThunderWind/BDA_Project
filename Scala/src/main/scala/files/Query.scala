package files

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.{Normalizer, StandardScaler, VectorAssembler}
import org.apache.spark.ml.linalg.{Vectors, Vector}

//import org.apache.commons.lang.mutable.Mutable
import org.apache.spark.SparkContext
import org.apache.spark.sql.{ColumnName, DataFrame, SparkSession, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.{col, regexp_replace}
import org.apache.spark.sql.functions.array_contains
import scala.collection.mutable.ArraySeq

import scala.collection.mutable.Map
import java.time.Instant
import scala.collection.mutable
import breeze.numerics.I

case class SongData(
                       timestamp: Instant,
                       artist_id: Types.artist_id,
                       artist_name: Types.artist_name,
                       artist_location: Types.artist_location,
                       artist_latitude: Types.artist_latitude,
                       artist_longitude: Types.artist_longitude,
                       song_id: Types.song_id,
                       title: Types.title,
                       song_hotness: Types.song_hotness,
                       similar_artists: Types.similar_artists, // à utliser pour évaluer clustering des artistes
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

case class ArtistData(
    artist_id: Types.artist_id,
    artist_name: Types.artist_name,
    artist_location: Types.artist_location,
    artist_latitude: Types.artist_latitude,
    artist_longitude: Types.artist_longitude,
    nbSong : Int,
    avgSongDuration: Float, //ou autre type
    avgSongLoudness: Float,
    avgTempo: Float,
    yearFirstSong: Types.year,
    yearLastSong: Types.year,
    avgLoudness: Float,
    avgEnergy: Float
    // ajouter d'autres colonnes ?
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
        /*
        // question 1: which year holds the highest number of produced tracks ?
        val groupedYear = groupByCount(data,"year").filter(col("year") =!= 0)
        // which country is home to the highest number of artists ?
        val groupedLocation = groupByCount(data,"artist_location")
        // which are the most popular music genre ?
        val groupedGenre = uniqueGenreCount(data, columnName = "artist_terms")

        groupedGenre.show(10)

        // question 2: what is the average BPM per music genre?
        val listGenre = groupedGenre.select("term").collect().map(_.getString(0)).toList
        val avgBPM = avgMetricbyGenre(data, "tempo", listGenre)
        val avgBPMResults: Unit = printResults(avgBPM, "tempo")
        // what is the average loudness per music genre?
        val avgLoudness = avgMetricbyGenre(data, "loudness", listGenre)
        val avgLoudnessResults: Unit = printResults(avgLoudness, "loudness")

        // question 4:
        // Dans une optique de recommandation d'un artiste à un utilisateur,
        // comment pourrait-on mesurer la similarité entre artistes ?
        */
        question4(data)
    }

    def question4(data: DataFrame): Unit = {
        val artistData = create_artist_dataframe(data)

        val test = get_similar_artists(data)
        val scaledData = preprocessData(artistData)
        val predictions = performKMeans(scaledData, 10)
        evaluateCluster(predictions, test, true)

        val ks = Array(5, 10, 20, 50, 100)
        val metrics = ks.map(k => (k, evaluateCluster(performKMeans(scaledData, k), test, false)))
        println(metrics.mkString("\n"))
    }

    def preprocessData(data: DataFrame): DataFrame = {
        // à partir de artistData, on veut filtrer les colonnes pour ne récupérer que celles qui contiennent des float
        val assembler = new VectorAssembler()
                    .setInputCols(Array("avgSongDuration", "avgSongLoudness", "avgTempo", "avgLoudness", "avgEnergy"))
                    .setOutputCol("features")

        val assembledData = assembler.transform(data)

        val scaler = new StandardScaler()
            .setInputCol("features")
            .setOutputCol("scaledFeatures")
            .setWithMean(true)
            .setWithStd(true)

        scaler.fit(assembledData).transform(assembledData)
    }

    def performKMeans(data: DataFrame, k: Int): DataFrame = {
        val kmeans = new KMeans().setK(k)

        val predictions = kmeans.fit(data).transform(data)
        predictions
    }

    def evaluateCluster(predictions: DataFrame, test: DataFrame, verbose:Boolean): Tuple4[Double, Double, Double, Double] = {
        val clusters = predictions.select("prediction", "artist_id").groupBy("prediction").agg(collect_list("artist_id").as("artist_ids"))
        val avgArtistsPerCluster = clusters.select(avg(size(col("artist_ids")))).first().getDouble(0)
        if (verbose) println("avgArtistsPerCluster: " + avgArtistsPerCluster)

        val evaluator = new ClusteringEvaluator()
        val silhouette = evaluator.evaluate(predictions)
        if (verbose) println("Silhouette with squared euclidean distance = " + silhouette)

        val joinedDF = clusters.join(predictions.select("artist_id","prediction"), Seq("prediction"))
        val removeIdFromSimilar = udf((artist_ids: Seq[String], artist_id: String) => {
            artist_ids.filterNot(_ == artist_id)
        })
        val pred = joinedDF.withColumn("similar", removeIdFromSimilar(col("artist_ids"), col("artist_id"))).select("artist_id", "similar")

        val joinedDF2 = test.join(pred, Seq("artist_id"))

        val calculatePercentage = udf((similarTest: Seq[String], similarPred: Seq[String]) => {
            val commonIds = similarTest.intersect(similarPred).distinct
            val percentage = (commonIds.length.toDouble / similarTest.length) * 100
            percentage
        })

        val resultDF = joinedDF2.withColumn("pourcentage", calculatePercentage(col("similar_artists"), col("similar")))
        val avgPercent = resultDF.select(avg(col("pourcentage"))).first().getDouble(0)
        val maxPercent = resultDF.select(max(col("pourcentage"))).first().getDouble(0)
        if (verbose) {
            println(s"avg accuracy: $avgPercent, max accuracy: $maxPercent")
        }
        Tuple4(avgArtistsPerCluster, silhouette, avgPercent, maxPercent)
    }

    def printNames(predictions: DataFrame): Unit = {
        val names = predictions.select("prediction", "artist_name").groupBy("prediction").agg(collect_list("artist_name").as("similar_artists"))
        val names2 = names.join(predictions.select("artist_name","prediction"), Seq("prediction"))
        names2.select("artist_name","similar_artists").show(10, false)
    }

    def create_artist_dataframe(data: DataFrame): DataFrame = {
        val artistData = data.select("artist_id", "artist_name", "artist_location", "artist_latitude", "artist_longitude", "duration", "energy", "loudness", "tempo", "year")
        val groupedArtist = artistData.groupBy("artist_id")
        val artist_infos = groupedArtist.agg(
            first("artist_name").as("artist_name"),
            count("artist_id").as("nbSong"),
            avg("duration").as("avgSongDuration"),
            avg("loudness").as("avgSongLoudness"),
            avg("tempo").as("avgTempo"),
            min("year").as("yearFirstSong"),
            max("year").as("yearLastSong"),
            avg("loudness").as("avgLoudness"),
            avg("energy").as("avgEnergy")
        )
        return artist_infos.na.drop()
    }

    def get_similar_artists(data: DataFrame): DataFrame = {
        val df = data.select("artist_id", "similar_artists").groupBy("artist_id").agg(first("similar_artists").as("similar_artists"))
        val parseStringList = spark.udf.register("parseStringList", (chaine: String) => {
            chaine.drop(1).dropRight(1).split(",").toList.map(_.drop(1).dropRight(1))
        })
        return df.withColumn("similar_artists", parseStringList(df("similar_artists")))
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
