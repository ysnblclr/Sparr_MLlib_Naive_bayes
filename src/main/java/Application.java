import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator$;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class Application {
    public static void main(String[] args) {


        SparkSession sparkSession = SparkSession.builder().master("local").appName("spark-mllib-naive-bayes").getOrCreate();

        Dataset<Row> loadData = sparkSession.read().format("csv").option("header", "true").option("inferSchema", "True").load("C:\\Users\\Msi\\Desktop\\verisetleri\\basketbol.csv");




        //data içindeki string ifadeleri modelleyebilmemiz için indexlememiz gerekiyor
        //2 parametre alıyoru girşi kolonu , çıkış kolonu
        StringIndexer indexHava = new StringIndexer().setInputCol("hava").setOutputCol("hava_cat");
        StringIndexer indexNem = new StringIndexer().setInputCol("nem").setOutputCol("nem_cat");
        StringIndexer indexSicaklik = new StringIndexer().setInputCol("sicaklik").setOutputCol("sicaklik_cat");
        StringIndexer indexRuzgar = new StringIndexer().setInputCol("ruzgar").setOutputCol("ruzgar_cat");

        //sonucu label olarak yazıyoruz. bulmak istediğimiz kolon
        StringIndexer indexLabel = new StringIndexer().setInputCol("basketbol").setOutputCol("label");




        //uygulamak içinse hava adında bir kolon bulursan gelen tabloda bunu fit ile bul sonra transdorm et
        //önceki dataseti bir sonraskinin girdisi oluyor
        Dataset<Row> transformHava = indexHava.fit(loadData).transform(loadData);
        Dataset<Row> transformNem = indexNem.fit(transformHava).transform(transformHava);
        Dataset<Row> transformSicaklik = indexSicaklik.fit(transformNem).transform(transformNem);
        Dataset<Row> transformRuzgar = indexRuzgar.fit(transformSicaklik).transform(transformSicaklik);
        Dataset<Row> transformResult = indexLabel.fit(transformRuzgar).transform(transformRuzgar);

        VectorAssembler vectorAssembler=new VectorAssembler().setInputCols(new String[]{"hava_cat","nem_cat","sicaklik_cat","ruzgar_cat","label"}).setOutputCol("features");

        Dataset<Row> transform = vectorAssembler.transform(transformResult);

        Dataset<Row> final_data = transform.select("label", "features");

        //train test ayırıyoruz
        Dataset<Row>[] datasets = final_data.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> train_data = datasets[0];
        Dataset<Row> test_data = datasets[1];

        NaiveBayes nb=new NaiveBayes();
        nb.setSmoothing(1);
        NaiveBayesModel model = nb.fit(train_data);
        Dataset<Row> prediction = model.transform(test_data);

       prediction.show();

        //modelin değerlendirmesi
        MulticlassClassificationEvaluator evalator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");// buradaki boşluktan dolayı hata almıştı :)
        //yaptığıız tahmini değerlendiriyoruz. double tahmin değeri döndürüyor
        double evaluate= evalator.evaluate(prediction);
        System.out.println("Accuracy= "+evaluate);

        //final_data.show();// yağmurluya 1 kapalıya 2 günrşliye 3 değerini vermiş




    }
}
