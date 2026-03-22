"""
Huấn luyện mô hình – Logistic Regression & Random Forest
Sử dụng MLlib của PySpark.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler, Imputer
from pyspark.sql import functions as F


TARGET = "TARGET"
SEED = 42


def get_spark(app_name: str = "HomeCreditModel") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def load_features(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.parquet(path)


def get_feature_cols(df: DataFrame) -> list[str]:
    exclude = {TARGET, "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"}
    return [
        f.name for f in df.schema.fields
        if f.name not in exclude
        and str(f.dataType) in ("DoubleType()", "FloatType()", "IntegerType()", "LongType()")
    ]


def build_pipeline(feature_cols: list[str], classifier) -> Pipeline:
    imputer = Imputer(inputCols=feature_cols, outputCols=[f + "_imp" for f in feature_cols])
    imputed_cols = [f + "_imp" for f in feature_cols]
    assembler = VectorAssembler(inputCols=imputed_cols, outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features",
                            withMean=True, withStd=True)
    return Pipeline(stages=[imputer, assembler, scaler, classifier])


def evaluate(predictions: DataFrame) -> dict:
    evaluator = BinaryClassificationEvaluator(labelCol=TARGET, rawPredictionCol="rawPrediction")
    auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
    return {"AUC-ROC": round(auc, 4)}


def train_and_evaluate(spark: SparkSession, features_path: str) -> None:
    df = load_features(spark, features_path)
    feature_cols = get_feature_cols(df)

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=SEED)

    models = {
        "LogisticRegression": LogisticRegression(
            labelCol=TARGET, featuresCol="features",
            maxIter=100, regParam=0.01
        ),
        "RandomForest": RandomForestClassifier(
            labelCol=TARGET, featuresCol="features",
            numTrees=100, maxDepth=6, seed=SEED
        ),
    }

    for name, clf in models.items():
        print(f"\n{'─'*50}\nTraining {name}…")
        pipeline = build_pipeline(feature_cols, clf)
        model = pipeline.fit(train_df)
        predictions = model.transform(test_df)
        metrics = evaluate(predictions)
        print(f"{name} metrics: {metrics}")
        model.write().overwrite().save(f"output/model_{name}")
        print(f"Model saved → output/model_{name}")


if __name__ == "__main__":
    spark = get_spark()
    train_and_evaluate(spark, features_path="output/final_features.parquet")
    spark.stop()
