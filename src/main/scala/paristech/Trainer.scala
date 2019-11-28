package paristech

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.{CountVectorizer, IDF, StopWordsRemover}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()

    import spark.implicits._

    // limiter le niveau des messages
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegardé précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    // ATTENTION : mofifier les variables path et path_data  en fonction de votre environnement
    val path = "/home/benezeth/Documents/MSBigData/INF729Spark/spark_project_kickstarter_2019_2020-master/"
    val path_data = path + "data/"


    // Chargement du DataFrame
    val df: DataFrame = spark.read.parquet(path_data + "prepared_trainingset")


    /*******************************************************************************
      *
      * Utilisation des données textuelles
      *
      ********************************************************************************/

    // Stage 1 : récupérer les mots des textes
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    val wordsData = tokenizer.transform(df)

    // Stage 2 : retirer les stop words
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("removed")

    val wordsData2 = remover.transform(wordsData)
    wordsData2.select("text", "tokens", "removed").show(5)

    // Stage 3 : computer la partie TF
    val cvModel = new CountVectorizer()
      .setInputCol("removed")
      .setOutputCol("feature")
      .fit(wordsData2)

    val wordsData3 = cvModel.transform(wordsData2)

    // Stage 4 : computer la partie IDF
    val idf = new IDF().setInputCol("feature").setOutputCol("tfidf")
    val idfModel = idf.fit(wordsData3)

    val wordsData4 = idfModel.transform(wordsData3)


    /*******************************************************************************
      *
      * Conversion des variables catégorielles en variables numériques
      *
      ********************************************************************************/

    // Stage 5 : convertir country2 en quantités numériques
    val indexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .fit(wordsData4)
    val wordsData5 = indexer.transform(wordsData4)


    // Stage 6 : convertir currency2 en quantités numériques
    val indexer2 = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .fit(wordsData5)
    val wordsData6 = indexer2.transform(wordsData5)

    // Stages 7 et 8: One-Hot encoder ces deux catégories
    val encoder = new OneHotEncoder()
      .setInputCol("country_indexed")
      .setOutputCol("country_onehot")

    val wordsData7 = encoder.transform(wordsData6)

    val encoder2 = new OneHotEncoder()
      .setInputCol("currency_indexed")
      .setOutputCol("currency_onehot")

    val wordsData8 = encoder2.transform(wordsData7)

    /*******************************************************************************
      *
      * Mettre les données sous une forme utilisable par Spark.ML
      *
      ********************************************************************************/

    // Stage 9 : assembler tous les features en un unique vecteur
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")

    val wordsData9 = assembler.transform(wordsData8)

    // Stage 10 : créer/instancier le modèle de classification
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(20)

    /*******************************************************************************
      *
      * Création du Pipeline
      *
      ********************************************************************************/
    val stages = Array(tokenizer, remover, cvModel, idf, indexer, indexer2, encoder,encoder2, assembler, lr)
    val pipeline = new Pipeline().setStages(stages)


    /*******************************************************************************
      *
      * Entraînement, test, et sauvegarde du modèle
      *
      ********************************************************************************/

    // Split des données en training et test sets
    val Array(training, test) = df.randomSplit(Array(0.9, 0.1), seed = 1966)

    // Entraînement du modèle
    val model = pipeline.fit(training)

    // Test du modèle
    val dfWithSimplePredictions = model.transform(test)

    dfWithSimplePredictions.groupBy("final_status", "predictions").count.show()

    // On instancie un evaluator
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("f1")
      .setLabelCol("final_status")
      .setPredictionCol("predictions")

    // Afficher le f1-score du modèle sur les données de test
    val score=evaluator.evaluate(dfWithSimplePredictions)
    println("Le f1-score est " + score)


    /*******************************************************************************
      *
      * Réglage des hyper-paramètres (a.k.a. tuning) du modèle
      *
      ********************************************************************************/

    // Grid-search
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(cvModel.minDF, Array(55.0, 75.0, 95.0))
      .build()


    //  Il faut instancier un TrainValidationSplit
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    // On entraine le modèle sur les points de la Grid
    val validationModel = trainValidationSplit.fit(training)

    // Test du modèle
    val dfWithPredictions = validationModel.transform(test)

    // // Afficher le f1-score du Best modèle sur les données de test
    val score2=evaluator.evaluate(dfWithPredictions)
    println("Le f1-score est " + score2)

    dfWithPredictions.groupBy("final_status","predictions").count.show()

    // Sauvegarder le modèle entraîné pour pouvoir le réutiliser plus tard

    validationModel.write.overwrite().save(path + "model")
















  }
}
