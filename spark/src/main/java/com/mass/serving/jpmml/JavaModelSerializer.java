package com.mass.serving.jpmml;

import javax.xml.transform.stream.StreamResult;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.dmg.pmml.PMML;
import org.jpmml.model.JAXBUtil;
import org.jpmml.sparkml.PMMLBuilder;

import java.io.File;
import java.io.FileOutputStream;

public class JavaModelSerializer {
    public void serializeModel(PipelineModel pipelineModel,
                               String modelSavePath,
                               Dataset<Row> transformedData) {
        PMML pmml = new PMMLBuilder(transformedData.schema(), pipelineModel).build();
        try {
            FileOutputStream output = new FileOutputStream(new File(modelSavePath));
            JAXBUtil.marshalPMML(pmml, new StreamResult(output));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
