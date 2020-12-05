name := "mmlspark"

version := "0.1"

scalaVersion := "2.11.11"

resolvers += "MMLSpark" at "https://mmlspark.azureedge.net/maven"
libraryDependencies += "com.microsoft.ml.spark" %% "mmlspark" % "1.0.0-rc3"