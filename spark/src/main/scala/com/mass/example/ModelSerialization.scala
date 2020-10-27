package com.mass.example

import com.mass.data.AnimeData
import com.mass.model.Classifier
import com.mass.serving.jpmml.{JavaModelSerializer => ModelSerializerJPmml}
import com.mass.serving.mleap.{ModelSerializer => ModelSerializerMLeap}
import com.mass.utils.Context

object ModelSerialization extends Context {
  def main(args: Array[String]): Unit = {
    var data = AnimeData.readData("/rating.csv", "/anime.csv")
    data = AnimeData.processData(data, filterNA = false, convertLabel = true)
    println(s"data length: ${data.count()}, " +
      s"user_count: ${data.select("user_id").distinct.count()}, " +
      s"anime_count: ${data.select("anime_id").distinct.count()}")

    val model = new Classifier(Some("mlp"))
    time(model.train(data, evaluate = false, debug = true), "Training")
    val transformedData = model.transform(data)
    transformedData.show(4, truncate = false)

    val jpmmlModelPath = "spark/src/main/resources/jpmml_model/jpmml_model.pmml"
    val jpmmlModelSerializer = new ModelSerializerJPmml()
    jpmmlModelSerializer.serializeModel(model.pipelineModel, jpmmlModelPath, transformedData)

    val mleapModelPath = "jar:file:/spark/src/main/resources/mleap_model/mleap_model.zip"
    val mleapModelSerializer = new ModelSerializerMLeap()
    mleapModelSerializer.serializeModel(model.pipelineModel, mleapModelPath, transformedData)
  }
}
