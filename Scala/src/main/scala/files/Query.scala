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
        */

        // question 3 : TODO

        // question 4:
        // Dans une optique de recommandation d'un artiste à un utilisateur,
        // comment pourrait-on mesurer la similarité entre artistes ? -> Machne learning

        // décrire chaque artiste par une liste de caractéristiques
        // faire du clustering pour essayer de regrouper les artistes similaires
        // utiliser algorithmes de clustering disponible dans scala mllib

        val artistData = create_artist_dataframe(data)
        //artistData.show(10)

        val test = get_similar_artists(data)

        //test.dtypes.foreach(println)
        test.show(10)


        // à partir de artistData, on veut filtrer les colonnes pour ne récupérer que celles qui contiennent des float
        val assembler = new VectorAssembler()
                    .setInputCols(Array("artist_latitude")) //, "artist_longitude", "avgSongDuration", "avgSongLoudness", "avgTempo", "avgLoudness", "avgEnergy"))
                    .setOutputCol("features")

        val assembledData = assembler.transform(artistData)

        val scaler = new StandardScaler()
            .setInputCol("features")
            .setOutputCol("scaledFeatures")
            .setWithMean(true)
            .setWithStd(true)

        val scaledData = scaler.fit(assembledData).transform(assembledData)

        // puis on veut faire du clustering sur ces colonnes

        val kmeans = new KMeans().setK(40)

        val predictions = kmeans.fit(scaledData).transform(scaledData)
        predictions.show(10)

        val clusters = predictions.select("prediction", "artist_id").groupBy("prediction").agg(collect_list("artist_id").as("artist_ids"))
        //clusters.show(10, false)

        // compute avg number of artists per cluster
        val avgArtistsPerCluster = clusters.select(avg(size(col("artist_ids")))).first().getDouble(0)
        println("avgArtistsPerCluster: " + avgArtistsPerCluster)

        // puis on veut évaluer la qualité du clustering
        val evaluator = new ClusteringEvaluator()

        val silhouette = evaluator.evaluate(predictions)
        println(s"Silhouette with squared euclidean distance = $silhouette")


        // créer pred et test :
        val joinedDF = clusters.join(predictions.select("artist_id","prediction"), Seq("prediction"))

        val removeIdFromSimilar = udf((artist_ids: Seq[String], artist_id: String) => {
            artist_ids.filterNot(_ == artist_id)
        })

        val pred = joinedDF.withColumn("similar", removeIdFromSimilar(col("artist_ids"), col("artist_id"))).select("artist_id", "similar")

        pred.show(10)

        // pour chaque artiste de similar, on regarde on regarde dans quel cluster il se trouve,
        // on récupre les artistes qui se trouvent dans le même cluster,
        // et on regarde le nombre d'artistes qui sont dans la liste des artistes similaires et qui sont aussi dans le même cluster
        // on fait ça pour chaque artiste de similar et on calcule le pourcentage d'artistes similaires trouvés
        // on fait la moyenne de ces pourcentages pour avoir un pourcentage moyen
        // pour savoir si notre prédiction de la similarité est correct

        val joinedDF2 = test.join(pred, Seq("artist_id"))

        val calculatePercentage = udf((similarTest: Seq[String], similarPred: Seq[String]) => {
            val commonIds = similarTest.intersect(similarPred).distinct
            val percentage = (commonIds.length.toDouble / similarTest.length) * 100
            percentage
        })

        val resultDF = joinedDF2.withColumn("pourcentage", calculatePercentage(col("similar_artists"), col("similar")))

        resultDF.show(10)

        resultDF.select(avg(col("pourcentage"))).show()
        resultDF.select(max(col("pourcentage"))).show()

        // */

        // Normalisation des vecteurs de caractéristiques
        /*
        val normalizer = new Normalizer()
        .setInputCol("features")
        .setOutputCol("normalizedFeatures")
        .setP(2.0)

        val normalizedData = normalizer.transform(assembledData)

        // Sélection des vecteurs normalisés
        val vectors = normalizedData.select("artist_id", "normalizedFeatures")
        .rdd
        .map { case row => (row.getAs[Int]("artist_id"), row.getAs[Vector]("normalizedFeatures")) }

        // Calcul de la similarité du cosinus
        def cosineSimilarity(v1: Vector, v2: Vector): Double = {
            val dotProduct = v1.dot(v2)
            val magnitude = Vectors.norm(v1,2) * Vectors.norm(v2,2)
            dotProduct / magnitude
        }

        val similarities = vectors.cartesian(vectors)
                .map { case ((id1, vector1), (id2, vector2)) =>
                    (id1, id2, cosineSimilarity(vector1, vector2))
                }

        // Affichage des similarités
        val df = spark.createDataFrame(similarities)
        df.show(10)
        // */
    }

    def create_artist_dataframe(data: DataFrame): DataFrame = {
        val artistData = data.select("artist_id", "artist_name", "artist_location", "artist_latitude", "artist_longitude", "duration", "energy", "loudness", "tempo", "year")
        val groupedArtist = artistData.groupBy("artist_id")
        val artist_infos = groupedArtist.agg(
            first("artist_name").as("artist_name"),
            first("artist_location").as("artist_location"),
            first("artist_latitude").as("artist_latitude"),
            first("artist_longitude").as("artist_longitude"),
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
