package paristech

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame

import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.datediff
import org.apache.spark.sql.functions.from_unixtime
import org.apache.spark.sql.functions.when
import org.apache.spark.sql.functions.concat_ws
import org.apache.spark.sql.functions.lower


object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()

    import spark.implicits._

    // limiter le niveau des messages
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)


    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    // ATTENTION : mofifier les variables path et path_data  en fonction de votre environnement
    val path = "/home/benezeth/Documents/MSBigData/INF729Spark/spark_project_kickstarter_2019_2020-master/"
    val path_data = path + "data/"

    //******************************************************************************
    // Chargement des données
    //******************************************************************************
    val df: DataFrame = spark
      .read
      .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .csv(path_data + "train_clean.csv")

    println(s"Nombre de lignes : ${df.count}")
    println(s"Nombre de colonnes : ${df.columns.length}")

    df.show()

    df.printSchema()

    val dfCasted: DataFrame = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline" , $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))

    dfCasted.printSchema()

    //******************************************************************************
    // Cleaning
    //******************************************************************************
    dfCasted
      .select("goal", "backers_count", "final_status")
      .describe()
      .show

    dfCasted.groupBy("disable_communication").count.orderBy($"count".desc).show(100)

    dfCasted.groupBy("country").count.orderBy($"count".desc).show(100)

    dfCasted.groupBy("currency").count.orderBy($"count".desc).show(100)

    dfCasted.select("deadline").dropDuplicates.show()

    dfCasted.groupBy("state_changed_at").count.orderBy($"count".desc).show(100)

    dfCasted.groupBy("backers_count").count.orderBy($"count".desc).show(100)

    dfCasted.select("goal", "final_status").show(30)

    dfCasted.groupBy("country", "currency").count.orderBy($"count".desc).show(50)


    val df2: DataFrame = dfCasted.drop("disable_communication")

    //******************************************************************************
    // Les fuites du futur
    //******************************************************************************
    val dfNoFutur: DataFrame = df2.drop("backers_count", "state_changed_at")

    //******************************************************************************
    // Colonnes currency et country
    //******************************************************************************
    df.filter($"country" === "False")
      .groupBy("currency")
      .count
      .orderBy($"count".desc)
      .show(50)

    def cleanCountry(country: String, currency: String): String = {
      if (country == "False")
        currency
      else
        country
    }

    def cleanCurrency(currency: String): String = {
      if (currency != null && currency.length != 3)
        null
      else
        currency
    }

    val cleanCountryUdf = udf(cleanCountry _)
    val cleanCurrencyUdf = udf(cleanCurrency _)

    val dfCountry: DataFrame = dfNoFutur
      .withColumn("country2", cleanCountryUdf($"country", $"currency"))
      .withColumn("currency2", cleanCurrencyUdf($"currency"))
      .drop("country", "currency")


    /* ou encore, en utilisant sql.functions.when:
    dfNoFutur
      .withColumn("country2", when($"country" === "False", $"currency").otherwise($"country"))
      .withColumn("currency2", when($"country".isNotNull && length($"currency") =!= 3, null).otherwise($"currency"))
      .drop("country", "currency")
    */

     //******************************************************************************
     //Affichez le nombre d’éléments de chaque classe (colonne final_status).
     //Conservez uniquement les lignes qui nous intéressent pour le modèle, à savoir lorsque final_status vaut 0 (Fail) ou 1 (Success).
    //******************************************************************************
    dfCountry.groupBy("final_status").count.orderBy($"count".desc).show(100)

    val dfCountry2: DataFrame = dfCountry
      .filter(($"final_status" === 0) || ($"final_status" === 1))


    //******************************************************************************
    // Ajouter et manipuler des colonnes
    //******************************************************************************

    val dfCountry_mod = dfCountry2
      .withColumn("days_campaign", datediff(from_unixtime($"deadline"), from_unixtime($"created_at")))
      .withColumn("hours_prepa", (($"launched_at" - $"created_at")/3.6).cast("Int")/1000)
      .drop($"launched_at")
      .drop($"created_at")
      .drop($"deadline")
      .withColumn("text", concat_ws(" ", lower($"name"), lower($"desc"), lower($"keywords")))
      .withColumn("country2", when($"country2" === null, "unknown").otherwise($"country2"))
      .withColumn("currency2", when($"currency2" === null, "unknown").otherwise($"currency2"))
      .withColumn("days_campaign", when($"days_campaign" === null, -1).otherwise($"days_campaign"))
      .withColumn("hours_prepa", when($"hours_prepa" === null, -1).otherwise($"hours_prepa"))
      .withColumn("goal", when($"goal" === null, -1).otherwise($"goal"))

    // on ne garde que les lignes avec une date de création antérieure à date de lancement
    val dfCountry_modfiltered = dfCountry_mod.filter($"hours_prepa" >= 0)


    //******************************************************************************
    // Sauvegarder un DataFrame
    //******************************************************************************
    dfCountry_modfiltered.write.mode("overwrite").parquet(path_data + "dfCountry_modfiltered")
  }
}
