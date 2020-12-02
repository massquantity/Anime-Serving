package com.mass.evaluate

import com.mass.data.DataSplitter.stratified_split
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.{GBTRegressor, GeneralizedLinearRegression}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.{Model, Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.DataFrame

import scala.util.Random

class EvalRegressor(algo: Option[String], pipelineStages: Array[PipelineStage]) {
  def eval(data: DataFrame): Unit = {
    val Array(trainData, evalData) = stratified_split(data, 0.8, "user_id")
    val pipeline = new Pipeline().setStages(pipelineStages)
    val pipeModel: PipelineModel = pipeline.fit(trainData)
    val trainTransformedData = pipeModel.transform(trainData)
    val evalTransformedData = pipeModel.transform(evalData)
    trainTransformedData.cache()
    evalTransformedData.cache()

    algo match {
      case Some("gbdt") => evaluateGBDT(trainTransformedData, evalTransformedData)
      case Some("glr") => evaluateGLR(trainTransformedData, evalTransformedData)
      case None =>
        println("Model muse either be MultilayerPerceptronClassifier or RandomForestClassifier")
        System.exit(1)
      case _ =>
        println("Model muse either be MultilayerPerceptronClassifier or RandomForestClassifier")
        System.exit(2)
    }
    trainTransformedData.unpersist()
    evalTransformedData.unpersist()
  }

  def evaluateGBDT(trainData: DataFrame, evalData: DataFrame): Unit = {
    val gbr = new GBTRegressor()
      .setSeed(Random.nextLong())
      .setFeaturesCol("featureVector")
      .setLabelCol("rating")
      .setPredictionCol("pred")

    val pipeline = new Pipeline().setStages(Array(gbr))
    val paramGrid = new ParamGridBuilder()
      .addGrid(gbr.featureSubsetStrategy, Seq("onethird", "all", "sqrt", "log2"))
      .addGrid(gbr.maxDepth, Seq(3, 5, 7, 8))
      .addGrid(gbr.stepSize, Seq(0.01, 0.03))
      .addGrid(gbr.subsamplingRate, Seq(0.6, 0.8, 1.0))
      .build()
    showScoreAndParam(trainData, evalData, pipeline, paramGrid)
  }

  def evaluateGLR(trainData: DataFrame, evalData: DataFrame): Unit = {
    val glr = new GeneralizedLinearRegression()
      .setFeaturesCol("featureVector")
      .setLabelCol("rating")
      .setPredictionCol("pred")

    val pipeline = new Pipeline().setStages(pipelineStages ++ Array(glr))
    val paramGrid = new ParamGridBuilder()
      .addGrid(glr.family, Seq("gaussian"))
      .addGrid(glr.link, Seq("identity"))
      .addGrid(glr.maxIter, Seq(20, 50))
      .addGrid(glr.regParam, Seq(0.0, 0.01, 0.1))
      .build()
    showScoreAndParam(trainData, evalData, pipeline, paramGrid)
  }

  private [evaluate] def showScoreAndParam(trainData: DataFrame,
                                           evalData: DataFrame,
                                           pipeline: Pipeline,
                                           params: Array[ParamMap]): Unit = {
    val regressorEval = new RegressionEvaluator()
      .setLabelCol("rating")
      .setPredictionCol("pred")
      .setMetricName("rmse")

    val validator = new TrainValidationSplit()
      .setSeed(Random.nextLong())
      .setEstimator(pipeline)
      .setEvaluator(regressorEval)
      .setEstimatorParamMaps(params)
      .setTrainRatio(0.8)

    val validatorModel = validator.fit(trainData)
    val paramsAndMetrics = validatorModel.validationMetrics
      .zip(validatorModel.getEstimatorParamMaps).sortBy(-_._1)

    println("scores and params: ")
    paramsAndMetrics.foreach { case (metricScore, parameters) =>
      println(s"$parameters \t => $metricScore")
      println()
    }
    println()

    val bestModel: Model[_] = validatorModel.bestModel
    println(s"best model params: ${bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap}"
      + s" => rmse: ${validatorModel.validationMetrics.min}")
    println(s"eval rmse: ${regressorEval.evaluate(bestModel.transform(evalData))}")
  }
}
