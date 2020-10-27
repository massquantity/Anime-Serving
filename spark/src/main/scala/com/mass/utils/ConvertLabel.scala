package com.mass.utils

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

object ConvertLabel {
  def convert(data: DataFrame): DataFrame = {
    val udfMapValue = udf(mapValue(_: Int): Int)
    data.withColumn("label", udfMapValue(col("rating")))
  }

  private def mapValue(rating: Int): Int = {
    rating match {
      case `rating` if rating >= 9 => 2
      case `rating` if rating >= 6 => 1
      case _ => 0
    }
  }
}
