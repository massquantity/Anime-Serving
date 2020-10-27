package com.mass.model

import com.mass.evaluate.EvalClassifier
import com.mass.feature.FeatureTransformer
import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier, RandomForestClassifier}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

import scala.util.Random

class Classifier(algo: Option[String] = Some("mlp")) extends Serializable {
  var pipelineModel: PipelineModel = _

  def train(data: DataFrame, evaluate: Boolean, debug: Boolean): Unit = {
    val prePipelineStages: Array[PipelineStage] = FeatureTransformer.preProcessPipeline(data)

    if (debug) {
      val pipeline = new Pipeline().setStages(prePipelineStages)
      pipelineModel = pipeline.fit(data)
      val transformed = pipelineModel.transform(data)
      transformed.show(4, truncate = false)
    }

    if (evaluate) {
      val evalModel = new EvalClassifier(algo, prePipelineStages)
      evalModel.eval(data)
    }
    else {
      algo match {
        case Some("mlp") =>
          val tempData = new Pipeline().setStages(prePipelineStages).fit(data).transform(data)
          val inputSize = tempData
            .select("featureVector")
            .first()
            .getAs[Vector]("featureVector")
            .size

          val model = new MultilayerPerceptronClassifier()
            .setFeaturesCol("featureVector")
            .setLabelCol("label")
            .setPredictionCol("pred")
            .setProbabilityCol("prob")
            .setLayers(Array(inputSize, 40, 10, 3))
            .setStepSize(0.01)
            .setMaxIter(10)
          val pipelineStages = prePipelineStages ++ Array(model)
          val pipeline = new Pipeline().setStages(pipelineStages)
          pipelineModel = pipeline.fit(data)

        case Some("rf") =>
          val model = new RandomForestClassifier()
            .setFeaturesCol("featureVector")
            .setLabelCol("label")
            .setPredictionCol("pred")
            .setProbabilityCol("prob")
            .setFeatureSubsetStrategy("auto")
            .setMaxDepth(3)
            .setNumTrees(100)
            .setSubsamplingRate(1.0)
            .setSeed(Random.nextLong())
          val pipelineStages = prePipelineStages ++ Array(model)
          val pipeline = new Pipeline().setStages(pipelineStages)
          pipelineModel = pipeline.fit(data)

        case _ =>
          println("Model muse either be MultilayerPerceptronClassifier or RandomForestClassifier")
          System.exit(1)
      }
    }
  }

  def transform(dataset: DataFrame): DataFrame = {
    pipelineModel.transform(dataset)
  }
}