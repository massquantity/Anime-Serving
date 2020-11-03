package com.xgboost;

import com.xgboost.DataLoader.DenseData;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static com.xgboost.DataLoader.loadCSVFile;

public class TrainXGB {
    public static void main(String[] args) throws IOException, XGBoostError {
        DenseData trainCSVData = loadCSVFile("/xgboost/train_data.csv");
        DenseData testCSVData = loadCSVFile("/xgboost/test_data.csv");

        DMatrix trainData = new DMatrix(trainCSVData.data, trainCSVData.nrow, trainCSVData.ncol, 0.0f);
        DMatrix testData = new DMatrix(testCSVData.data, testCSVData.nrow, testCSVData.ncol, 0.0f);
        trainData.setLabel(trainCSVData.labels);
        testData.setLabel(testCSVData.labels);

        Map<String, Object> params = new HashMap<>();
        params.put("max_depth", 3);
        params.put("eta", 0.2);
        params.put("min_child_weight", 1);
        params.put("objective", "reg:squarederror");
        params.put("silent", 0);

        Map<String, DMatrix> watches = new HashMap<>();
        watches.put("train", trainData);
        watches.put("test", testData);

        int earlyStoppingRound = 2;
        int num_round = 5;
        float[][] eval_matrics = new float[watches.size()][num_round];
        Booster booster = XGBoost.train(
                trainData, params, num_round, watches, eval_matrics, null, null, earlyStoppingRound
        );

        float[][] predicts = booster.predict(testData);
        System.out.println("train matrics, test metrics: " + Arrays.deepToString(eval_matrics));

        //save model to modelPath
        String xgbPath = "/xgboost/model";
        File file = new File(xgbPath);
        if (!file.exists()) {
            file.mkdirs();
        }

        booster.saveModel(xgbPath + "/xgb.model");
        //dump model with feature map
        String[] modelInfos = booster.getModelDump(xgbPath + "/featmap.txt", false);
        saveDumpModel(xgbPath + "/dump.raw.txt", modelInfos);

        //save dmatrix into binary buffer
        testData.saveBinary(xgbPath + "/dtest.buffer");

        //reload model and data
        Booster booster2 = XGBoost.loadModel(xgbPath + "/xgb.model");
        DMatrix testMat2 = new DMatrix(xgbPath + "/dtest.buffer");
        float[][] predicts2 = booster2.predict(testMat2);

        //check the two predicts
        System.out.println(checkPredicts(predicts, predicts2));
    }

    public static boolean checkPredicts(float[][] fPredicts, float[][] sPredicts) {
        if (fPredicts.length != sPredicts.length) {
            System.out.println("length doesn't match");
            return false;
        }

        for (int i = 0; i < fPredicts.length; i++) {
            if (!Arrays.equals(fPredicts[i], sPredicts[i])) {
                return false;
            }
        }
        return true;
    }

    public static void saveDumpModel(String modelPath, String[] modelInfos) {
        try {
            PrintWriter writer = new PrintWriter(modelPath, "UTF-8");
            for (int i = 0; i < modelInfos.length; i++) {
                writer.print("booster[" + i + "]:\n");
                writer.print(modelInfos[i]);
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
