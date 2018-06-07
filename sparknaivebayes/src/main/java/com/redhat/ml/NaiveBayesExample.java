package com.redhat.ml;

import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.text.DecimalFormat;

/**
 * Created by samueltauil on 5/31/18.
 */
public class NaiveBayesExample {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("NaiveBayesExample")
                .master("redhatml")
                .getOrCreate();

        String masterServiceName = System.getenv("SPARK_MASTER");
        System.out.println(masterServiceName);
        spark.sparkContext().setLogLevel("ERROR");


        // Load training data
        Dataset<Row> dataFrame =
                spark.read().format("libsvm").load("src/main/resources/iris_svm.txt");
        // Split the data into train and test
        Dataset<Row>[] splits = dataFrame.randomSplit(new double[]{0.6, 0.4}, 1234L);
        Dataset<Row> train = splits[0];
        Dataset<Row> test = splits[1];

        // create the trainer and set its parameters
        NaiveBayes nb = new NaiveBayes();

        // train the model
        NaiveBayesModel model = nb.fit(train);

        // Select example rows to display.
        Dataset<Row> predictions = model.transform(test);
        predictions.show(200);

        // compute accuracy on the test set
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        accuracy *= 100;
        DecimalFormat df = new DecimalFormat("0.000");

        System.out.println("Test set accuracy = " + df.format(accuracy) + "%");

        spark.stop();
    }
}
