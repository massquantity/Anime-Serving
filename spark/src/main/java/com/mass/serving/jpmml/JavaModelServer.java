package com.mass.serving.jpmml;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.*;
import org.jpmml.model.PMMLUtil;
import org.xml.sax.SAXException;

import javax.xml.bind.JAXBException;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class JavaModelServer {
    private String modelPath;
    private Evaluator model;

    public JavaModelServer(String modelPath) {
        this.modelPath = modelPath;
    }

    private void loadModel() {
        InputStream ism = null;
        try {
            ism = new FileInputStream(new File(modelPath));
            this.model = new LoadingModelEvaluatorBuilder()
                    .load(ism)
                    .build();
            this.model.verify();

            List<? extends InputField> inputFields = model.getInputFields();
            for (InputField inputField : inputFields) {
                System.out.println(inputField.getName().getValue());
            }
        } catch (IOException | SAXException | JAXBException e) {
            System.err.println(e);
        } finally {
            try {
                assert ism != null;
                ism.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public Map<FieldName, ?> predict(Map<String, ?> featureMap) {
        if (model == null) loadModel();
        if (featureMap == null) {
            System.err.println("features are null");
            return null;
        }

        List<? extends InputField> inputFields = model.getInputFields();
        Map<FieldName, FieldValue> pmmlFeatureMap = new LinkedHashMap<>();
        for (InputField inputField : inputFields) {
            if (featureMap.containsKey(inputField.getName().getValue())) {
                Object value = featureMap.get(inputField.getName().getValue());
                pmmlFeatureMap.put(inputField.getName(), inputField.prepare(value));
            } else {
                System.err.println("lack of feature: " + inputField.getName().getValue());
                return null;
            }
        }
        return model.evaluate(pmmlFeatureMap);
    }
}
