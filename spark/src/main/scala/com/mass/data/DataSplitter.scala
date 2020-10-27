package com.mass.data

import com.mass.utils.Context
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions.{collect_list, row_number, size, rand}
import org.apache.spark.sql.expressions.Window
import scala.collection.mutable.ArrayBuffer

import scala.util.Random

object DataSplitter extends java.io.Serializable with Context {
  import spark.implicits._

  def parseRating(sep: String)(line: String): (Int, Int, Int) = {
    val features = line.split(sep)
    assert(features.size == 3)
    (features(0).toInt, features(1).toInt, features(2).toInt)
  }

  def randomSplit(df: DataFrame,
                  trainFrac: Double,
                  seed: Long = Random.nextLong()): Array[Dataset[Row]] = {
    df.randomSplit(Array(trainFrac, 1.0 - trainFrac), seed)
  }

  // stratified random split data according to each user
  def stratified_split(df: DataFrame,
                       trainFrac: Double,
                       userCol: String,
                       seed: Long = Random.nextLong()): Array[Dataset[Row]] = {

    val windowCount = Window.partitionBy(userCol)
    val windowSpec = Window.partitionBy(userCol).orderBy(rand(seed))
    val userCount = df.withColumn("count", size(collect_list("anime_id").over(windowCount)))
    val userRatio = userCount.withColumn("ratio", row_number().over(windowSpec) / $"count")

    val train_test = ArrayBuffer[DataFrame]()
    val ratioIndex = Array(trainFrac, 1.0)
    ratioIndex.zipWithIndex.foreach { case (ratio, i) =>
      if (i == 0)
        train_test += userRatio.filter($"ratio" <= ratio)
      else
        train_test += userRatio.filter($"ratio" <= ratio && $"ratio" > trainFrac)
    }
    Array(train_test(0), train_test(1))
  }

  // stratified split data according to each user in a chronological manner
  def stratified_chrono_split(df: DataFrame,
                              trainFrac: Double,
                              userCol: String): Array[Dataset[Row]] = {

    val windowCount = Window.partitionBy(userCol)
    val windowSpec = Window.partitionBy(userCol).orderBy($"timestamp".asc_nulls_last)
    val userCount = df.withColumn("count", size(collect_list("timestamp").over(windowCount)))
    val userRatio = userCount.withColumn("ratio", row_number().over(windowSpec) / $"count")

    val train_test = ArrayBuffer[DataFrame]()
    val ratioIndex = Array(trainFrac, 1.0)
    ratioIndex.zipWithIndex.foreach { case (ratio, i) =>
      if (i == 0)
        train_test += userRatio.filter($"ratio" <= ratio)
      else
        train_test += userRatio.filter($"ratio" <= ratio && $"ratio" > trainFrac)
    }
    Array(train_test(0), train_test(1))
  }
}
