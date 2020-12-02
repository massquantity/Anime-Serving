package com.mass.model

import com.mass.evaluate.EvalRegressor
import com.mass.feature.FeatureTransformer.preProcessPipeline
import org.apache.spark.ml.regression.{GBTRegressor, GeneralizedLinearRegression}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.DataFrame

import scala.util.Random

class Regressor(algo: Option[String] = Some("gbdt")) extends Serializable {
  var pipelineModel: PipelineModel = _

  def train(data: DataFrame, evaluate: Boolean, debug: Boolean): Unit = {
    val prePipelineStages: Array[PipelineStage] = preProcessPipeline(data)

    if (debug) {
      val pipeline = new Pipeline().setStages(prePipelineStages)
      pipelineModel = pipeline.fit(data)
      val transformed = pipelineModel.transform(data)
      transformed.show(4, truncate = false)
    }

    if (evaluate) {
      val evalModel = new EvalRegressor(algo, prePipelineStages)
      evalModel.eval(data)
    } else {
      algo match {
        case Some("gbdt") =>
          val model = new GBTRegressor()
            .setFeaturesCol("featureVector")
            .setLabelCol("rating")
            .setPredictionCol("pred")
            .setFeatureSubsetStrategy("auto")
            .setMaxDepth(3)
            .setMaxIter(5)
            .setStepSize(0.1)
            .setSubsamplingRate(0.8)
            .setSeed(2020L)
          val pipelineStages = prePipelineStages ++ Array(model)
          val pipeline = new Pipeline().setStages(pipelineStages)
          pipelineModel = pipeline.fit(data)

        case Some("glr") =>
          val model = new GeneralizedLinearRegression()
            .setFeaturesCol("featureVector")
            .setLabelCol("rating")
            .setPredictionCol("pred")
            .setFamily("gaussian")
            .setLink("identity")
            .setRegParam(0.0)
          val pipelineStages = prePipelineStages ++ Array(model)
          val pipeline = new Pipeline().setStages(pipelineStages)
          pipelineModel = pipeline.fit(data)

        case _ =>
          println("Model muse be GBDTRegressor or GeneralizedLinearRegression")
          System.exit(1)
      }
    }
  }

  def transform(dataset: DataFrame): DataFrame = {
    pipelineModel.transform(dataset)
  }

}
