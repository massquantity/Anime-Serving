package com.mass.data

import com.mass.feature.MultiHotEncoder
import com.mass.utils.{Context, ConvertLabel}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, Word2Vec}
import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.functions.{explode, mean}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StringType, StructField, StructType}

object AnimeData extends Context {
  import spark.implicits._

  def readData(ratingFile: String, animeFile: String): DataFrame = {
    val ratingSchema = new StructType(Array(
      StructField("user_id", IntegerType, nullable = false),
      StructField("anime_id", IntegerType, nullable = false),
      StructField("rating", IntegerType, nullable = false)
    ))
    val animeSchema = new StructType(Array(
      StructField("anime_id", IntegerType, nullable = false),
      StructField("name", StringType, nullable = true),
      StructField("genre", StringType, nullable = true),
      StructField("type", StringType, nullable = true),
      StructField("episodes", IntegerType, nullable = true),
      StructField("rating", DoubleType, nullable = true),
      StructField("members", IntegerType, nullable = true)
    ))

    val ratingPath = this.getClass.getResource(ratingFile).toString
    val animePath = this.getClass.getResource(animeFile).toString
    val rating = spark.read
      .option("inferSchema", "true")
      .option("header", "true")
      .schema(ratingSchema)
      .csv(ratingPath)
      .filter($"rating" =!= -1)
      .sample(withReplacement = false, fraction = 0.01, seed = 222)   // sampled !!!
    val anime = spark.read
      .option("inferSchema", "true")
      .option("header", "true")
      .schema(animeSchema)
      .csv(animePath)
      .withColumnRenamed("rating", "web_rating")
      .drop($"rating")
    rating.join(anime, Seq("anime_id"), "inner")
  }

  /**
    * Preprocess the data before saving
    * @param rawData original data
    * @param filterNA whether to filter missing values
    * @return
    */
  def processData(rawData: DataFrame, filterNA: Boolean, convertLabel: Boolean): DataFrame = {
    if (filterNA) {
      // println(s"find and fill NAs for each column...")
      // data.columns.foreach(x => println(s"$x -> ${data.filter(data(x).isNull).count}"))
      // data = data.filter(data.columns.map(data(_).isNotNull).reduce(_ && _))
      // data.columns is Array[String] type, suppose col is a column name, data(col) is Column type
      val allCols: Array[Column] = rawData.columns.map(rawData(_))
      val nullFilter: Column = allCols.map(_.isNotNull).reduce(_ && _)
      rawData.select(allCols: _*).filter(nullFilter)
    } else {
      rawData
        .na.fill("Missing", Seq("genre"))
        .na.fill("Missing", Seq("type"))
        .na.fill(rawData.selectExpr("mean(episodes) as mean").first.getAs[Double]("mean").toInt, Seq("episodes"))
        .na.fill(rawData.stat.approxQuantile("web_rating", Array(0.5), 0.1)(0), Seq("web_rating"))  // fill NA with median
      // .na.fill(rawData.selectExpr("mean(web_rating) as mean").first.getAs[Double]("mean"), Seq("web_rating"))
    }

    if (convertLabel) {
      ConvertLabel.convert(rawData)
    } else {
      rawData
    }
  }

  def transformText(df: DataFrame): Unit = {
    val regexTokenizer = new RegexTokenizer(uid = "regex_tokenizer")
      .setInputCol("name")
      .setOutputCol("words")
      .setPattern("\\W+")
      .setToLowercase(true)
      .setGaps(true)
      .setMinTokenLength(0)

    // val stopWords = new StopWordsRemover(uid = "stop_words")
    //  .setInputCol("words")
    //  .setOutputCol("st_words")
    //  .setCaseSensitive(false)

    val word2Vec = new Word2Vec(uid = "word2vec")
      .setInputCol("words")
      .setOutputCol("word_vectors")
      .setMinCount(0)
      .setVectorSize(16)
      .setSeed(2020L)

    val pipeline = new Pipeline().setStages(Array(regexTokenizer, word2Vec))
    val pipeModel: PipelineModel = pipeline.fit(df)
    val res = pipeModel.transform(df).select("name", "words", "word_vectors")
    res.filter($"name" === "Death Note").show(4, truncate = false)
    res.show(10, truncate = false)
  //  res.select("word_vectors").rdd.saveAsTextFile("spark/src/main/resources/wordsVec")
  }

  def multiHotData(df: DataFrame, output: Boolean): Unit = {
    val genreList: Array[String] = df
      .select("genre")
      .rdd
      .map(_.getAs[String]("genre"))
      .flatMap(_.split(","))
      .map(_.trim.replaceAll("\\s+", "").toLowerCase)
      .distinct
      .collect()

    println(s"total genre: ${genreList.length}")
    // genreList.foreach(println)

    val multihot = new MultiHotEncoder(uid = "multi_hot_encoder")
      .setInputCol("genre")
      .setOutputCols(genreList)
    val pipeline = new Pipeline().setStages(Array(multihot))
    val pipelineModel = pipeline.fit(df)
    val transData = pipelineModel.transform(df)
    transData.show(4, truncate = false)

    if (output) {
      transData
        .drop("splitted_genre")
        .coalesce(1)
        .write
        .format("parquet")
        .mode("overwrite")
        .option("header", "true")
        .save("spark/src/main/resources/multiHot22.parquet")
    }
  }
}
