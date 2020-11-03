package com.xgboost;

import org.apache.commons.lang3.ArrayUtils;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class DataLoader {
    public static DenseData loadCSVFile(String filePath) throws IOException {
        DenseData denseData = new DenseData();

        File f = new File(filePath);
        FileInputStream in = new FileInputStream(f);
        BufferedReader reader = new BufferedReader(new InputStreamReader(in, "UTF-8"));

        denseData.nrow = 0;
        denseData.ncol = -1;
        String line;
        List<Float> tlabels = new ArrayList<>();
        List<Float> tdata = new ArrayList<>();

        while ((line = reader.readLine()) != null) {
            String[] items = line.trim().split(",");
            if (items.length == 0) continue;
            denseData.nrow++;
            if (denseData.ncol == -1) {
                denseData.ncol = items.length - 1;
            }

            tlabels.add(Float.valueOf(items[0]));
            for (int i = 1; i < items.length; i++) {
                tdata.add(Float.valueOf(items[i]));
            }
        }

        reader.close();
        in.close();

        denseData.labels = ArrayUtils.toPrimitive(tlabels.toArray(new Float[0]));
        denseData.data = ArrayUtils.toPrimitive(tdata.toArray(new Float[0]));
        return denseData;
    }

    public static class DenseData {
        public float[] labels;
        public float[] data;
        public int nrow;
        public int ncol;
    }
}
