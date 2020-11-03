package com.xgboost

import ml.dmlc.xgboost4j.scala.spark.{XGBoostRegressionModel, XGBoostRegressor}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.SparkSession

object TrainXGBSpark {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("com").setLevel(Level.ERROR)

    val sparkConf: SparkConf = new SparkConf()
      .setAppName("XGBoost")
      .setMaster("local[*]")

    val spark: SparkSession = SparkSession
      .builder()
      .config(sparkConf)
      .getOrCreate()
    import spark.implicits._

    var trainData = spark.read
      .option("inferSchema", "true")
      .option("header", "false")
      .csv("/xgboost/train_data.csv")

    var testData = spark.read
      .option("inferSchema", "true")
      .option("header", "false")
      .csv("/xgboost/test_data.csv")
    trainData = trainData.withColumnRenamed("_c0", "label")
    testData = testData.withColumnRenamed("_c0", "label")
    // trainData.withColumn("label", $"_c0".cast("int"))
    // trainData.printSchema()

    trainData.show(4, truncate = false)
    val vectorAssembler = new VectorAssembler()
      .setInputCols(trainData.columns)
      .setOutputCol("feature")
    trainData = vectorAssembler.transform(trainData).select("feature", "label")
    testData = vectorAssembler.transform(testData).select("feature", "label")

    val xgbParam = Map(
      "max_depth" -> 5,
      "eta" -> 0.2,
      "min_child_weight" -> 1,
      "objective" -> "reg:squarederror",
      "num_round" -> 20,
      "allow_non_zero_for_missing" -> true,
      "maximize_evaluation_metrics" -> false,
      "eval_sets" -> Map("train" -> trainData, "test" -> testData)
    )
    val xgb = new XGBoostRegressor(xgbParam)
      .setFeaturesCol("feature")
      .setLabelCol("label")
      .setPredictionCol("pred")
      .setNumEarlyStoppingRounds(2)
      .setNumWorkers(4)
    val xgbRegressionModel: XGBoostRegressionModel = xgb.fit(trainData)
    val trainResult = xgbRegressionModel.transform(trainData)
    val testResult = xgbRegressionModel.transform(testData)
    testResult.select("pred", "label").show(4, truncate = false)

    val singleInstance = testResult.head().getAs[Vector]("feature")
    println(s"single prediciton: ${xgbRegressionModel.predict(singleInstance)}")

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setPredictionCol("pred")
      .setLabelCol("label")
    println(s"train rmse: ${evaluator.evaluate(trainResult)}, " +
      s"test rmse: ${evaluator.evaluate(testResult)}")
  }
}
