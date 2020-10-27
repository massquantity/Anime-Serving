package com.mass.feature

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCols}
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.sql.functions.{array_contains, col, split, trim, lower, regexp_replace}
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.util.Identifiable

import scala.collection.mutable.ArrayBuffer


class MultiHotEncoder(override val uid: String) extends Transformer
  with HasInputCol with HasOutputCols with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("multiHotEncoder"))

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCols(value: Array[String]): this.type = set(outputCols, value)

  override def transformSchema(schema: StructType): StructType = {
    var outputFields = ArrayBuffer[StructField]()
    schema.fields.foreach(f => outputFields += f)
    $(outputCols).foreach(o => outputFields += StructField(o, IntegerType, nullable = false))
    StructType(outputFields.toArray)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    var data = dataset.withColumn(
      "splitted_genre", split(lower(regexp_replace(col($(inputCol)), "\\s+", "")), "\\,")
    )
    $(outputCols).foreach { colName =>
      data = data.withColumn("genre_" + colName, array_contains(col("splitted_genre"), colName).cast("int"))
    }
    data.toDF()
  }

  override def copy(extra: ParamMap): MultiHotEncoder = defaultCopy(extra)
}


object MultiHotEncoder {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("com").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("MultiHotEncoder")

    val spark = SparkSession
      .builder
      .config(conf)
      .getOrCreate()
    import spark.implicits._

    val data = spark.read
      .option("inferSchema", "true")
      .option("header", "true")
      .csv(s"${this.getClass.getResource("/anime.csv").toString}")

  //  data
  //    .withColumn("array_genre_reg", split(lower(regexp_replace($"genre", "\\s+", "")), "\\,"))
  //    .show(4, truncate = false)

    val genreList: Array[String] = data
      .select("genre")
      .na.fill("Missing")
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
    val pipelineModel = pipeline.fit(data)
    pipelineModel.transform(data).show(4, truncate = false)
  }
}

