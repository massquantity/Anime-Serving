import com.microsoft.ml.spark.lightgbm.{LightGBMRegressionModel, LightGBMRegressor}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.SparkSession


object train_lightgbm {
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

    val par_path = "/lightgbm"

    val rating = spark.read
      .option("inferSchema", "true")
      .option("header", "true")
      .csv(par_path + "/rating.csv")
      .filter($"rating" =!= -1)

    val anime = spark.read
      .option("inferSchema", "true")
      .option("header", "true")
      .csv(par_path + "/anime.csv")
      .withColumnRenamed("rating", "web_rating")
      .withColumnRenamed("anime_id", "id")
      .drop("name")
      .drop("genre")

    var data = rating.join(anime, rating.col("anime_id") === anime.col("id"), "inner")

    data = data
      .drop("id")
      .withColumn("user_id", $"user_id".cast("string"))
      .withColumn("anime_id", $"anime_id".cast("string"))
      .withColumn("episodes", $"episodes".cast("int"))

    for (column <- Array("user_id", "anime_id", "type")) {
      val newCol = column + "Index"
      val encoder = new StringIndexer()
        .setInputCol(column)
        .setOutputCol(newCol)
        .setHandleInvalid("skip")
      data = encoder.fit(data).transform(data)
      data = data.withColumn(column, col(newCol).cast("int")).drop(newCol)
    }

    // data.show(4, truncate = false)
    // data.printSchema()

    var Array(trainData, evalData) = data.randomSplit(Array(0.8, 0.2), 0L)
    val vectorAssembler = new VectorAssembler()
      .setInputCols(trainData.columns.filter(_ != "rating"))
      .setOutputCol("feature")
      .setHandleInvalid("skip")
    trainData = vectorAssembler.transform(trainData) //.select("feature", "rating")
    evalData = vectorAssembler.transform(evalData) //.select("feature", "rating")

    val lgb = new LightGBMRegressor()
      .setObjective("regression")
      .setBoostingType("gbdt")
      .setMetric("rmse")
      .setLabelCol("rating")
      .setFeaturesCol("feature")
      .setPredictionCol("pred")
      .setLearningRate(1.0)
      .setNumLeaves(77)
      .setMinDataInLeaf(5)
      .setMaxDepth(0)
      .setMinSumHessianInLeaf(0.0)
      .setNumIterations(10)
      .setEarlyStoppingRound(5)
      .setCategoricalSlotNames(Array("user_id", "anime_id", "type"))
      .setVerbosity(1)

    val lgbModel: LightGBMRegressionModel = lgb.fit(trainData)
    val trainResult = lgbModel.transform(trainData)
    val evalResult = lgbModel.transform(evalData)

    println(lgbModel.getModel.model)
    evalResult.select("pred", "rating").show(4, truncate = false)

    val singleInstance = evalResult.head().getAs[Vector]("feature")
    println(s"single prediciton: ${lgbModel.predict(singleInstance)}")

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setPredictionCol("pred")
      .setLabelCol("rating")
    println(s"train rmse: ${evaluator.evaluate(trainResult)}, " +
      s"eval rmse: ${evaluator.evaluate(evalResult)}")
  }
}
