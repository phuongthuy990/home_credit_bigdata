"""
Thành viên 1 – Làm sạch dữ liệu (Data Cleaning)
Nhiệm vụ: xử lý giá trị null, phát hiện và loại bỏ anomaly.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F


def get_spark(app_name: str = "HomeCreditCleaning") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .getOrCreate()
    )


def load_csv(spark: SparkSession, path: str) -> DataFrame:
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(path)
    )


def drop_high_null_columns(df: DataFrame, threshold: float = 0.5) -> DataFrame:
    """Loại bỏ các cột có tỉ lệ null vượt ngưỡng threshold."""
    total = df.count()
    null_ratios = {
        col: df.filter(F.col(col).isNull()).count() / total
        for col in df.columns
    }
    cols_to_keep = [col for col, ratio in null_ratios.items() if ratio < threshold]
    return df.select(cols_to_keep)


def fill_nulls(df: DataFrame) -> DataFrame:
    """Điền null: median cho số, 'Unknown' cho chuỗi."""
    from pyspark.sql.types import StringType, NumericType

    for field in df.schema.fields:
        col = field.name
        if isinstance(field.dataType, StringType):
            df = df.fillna({col: "Unknown"})
        elif isinstance(field.dataType, NumericType):
            median_val = df.approxQuantile(col, [0.5], 0.01)[0]
            df = df.fillna({col: median_val})
    return df


def remove_anomalies(df: DataFrame) -> DataFrame:
    """
    Loại bỏ anomaly rõ ràng trong bộ dữ liệu Home Credit:
    - DAYS_BIRTH không âm  →  giá trị không hợp lệ
    - DAYS_EMPLOYED == 365243  →  mã hoá giá trị thiếu
    """
    if "DAYS_BIRTH" in df.columns:
        df = df.filter(F.col("DAYS_BIRTH") < 0)
    if "DAYS_EMPLOYED" in df.columns:
        df = df.withColumn(
            "DAYS_EMPLOYED",
            F.when(F.col("DAYS_EMPLOYED") == 365243, None)
             .otherwise(F.col("DAYS_EMPLOYED"))
        )
    return df


def clean_application(spark: SparkSession, input_path: str, output_path: str) -> DataFrame:
    df = load_csv(spark, input_path)
    df = drop_high_null_columns(df)
    df = remove_anomalies(df)
    df = fill_nulls(df)
    df.write.mode("overwrite").parquet(output_path)
    print(f"Cleaned data saved to {output_path} — rows: {df.count()}")
    return df


if __name__ == "__main__":
    spark = get_spark()
    clean_application(
        spark,
        input_path="data/application_train.csv",
        output_path="output/cleaned_data/application_train",
    )
    spark.stop()
