package com.mass.evaluate

import com.mass.data.DataSplitter.stratified_split
import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier, RandomForestClassifier}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.param.ParamMap

import scala.util.Random

class EvalClassifier(algo: Option[String], pipelineStages: Array[PipelineStage]) {
  def eval(data: DataFrame): Unit = {
    val Array(trainData, evalData) = stratified_split(data, 0.8, "user_id")
    val pipeline = new Pipeline().setStages(pipelineStages)
    val pipeModel: PipelineModel = pipeline.fit(trainData)
    val trainTransformedData = pipeModel.transform(trainData)
    val evalTransformedData = pipeModel.transform(evalData)
    trainTransformedData.cache()
    evalTransformedData.cache()

    algo match {
      case Some("mlp") => evaluateMLP(trainTransformedData, evalTransformedData)
      case Some("rf") => evaluateRF(trainTransformedData, evalTransformedData)
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

  private def evaluateMLP(trainData: DataFrame, evalData: DataFrame): Unit = {
    val inputSize = trainData
      .select("featureVector")
      .first()
      .getAs[Vector]("featureVector")
      .size

    val mlp = new MultilayerPerceptronClassifier()
      .setSeed(Random.nextLong())
      .setFeaturesCol("featureVector")
      .setLabelCol("label")
      .setPredictionCol("pred")
      .setProbabilityCol("prob")

    val pipeline = new Pipeline().setStages(Array(mlp))
    val paramGrid = new ParamGridBuilder()
      .addGrid(mlp.layers, Seq(Array[Int](inputSize, 20, 10, 3), Array[Int](inputSize, 50, 30, 10, 3)))
      .addGrid(mlp.stepSize, Seq(0.01, 0.03, 0.05))
      .addGrid(mlp.maxIter, Seq(10, 30, 50))
      .build()
    showScoreAndParam(trainData, evalData, pipeline, paramGrid)
  }

  private def evaluateRF(trainData: DataFrame, evalData: DataFrame): Unit = {
    val rf = new RandomForestClassifier()
      .setSeed(Random.nextLong())
      .setFeaturesCol("featureVector")
      .setLabelCol("label")
      .setPredictionCol("pred")
      .setProbabilityCol("prob")

    val pipeline = new Pipeline().setStages(Array(rf))
    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.featureSubsetStrategy, Seq("all", "sqrt", "log2"))
      .addGrid(rf.maxDepth, Seq(3, 5, 7, 8))
      .addGrid(rf.numTrees, Seq(20, 50, 100))
      .addGrid(rf.subsamplingRate, Seq(0.8, 1.0))
      .build()
    showScoreAndParam(trainData, evalData, pipeline, paramGrid)
  }

  private [evaluate] def showScoreAndParam(trainData: DataFrame,
                                           evalData: DataFrame,
                                           pipeline: Pipeline,
                                           params: Array[ParamMap]): Unit = {
    val classifierEval = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("pred")
      .setMetricName("accuracy")  // f1, weightedPrecision

    val validator = new TrainValidationSplit()
      .setSeed(Random.nextLong())
      .setEstimator(pipeline)
      .setEvaluator(classifierEval)
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

    val bestModel = validatorModel.bestModel
    println(s"best model params: ${bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap}" +
      s" => accuracy: ${validatorModel.validationMetrics.max}")
    println(s"eval accuracy: ${classifierEval.evaluate(bestModel.transform(evalData))}")
  }
}
