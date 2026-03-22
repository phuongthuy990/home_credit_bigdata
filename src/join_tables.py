"""
Thành viên 3 – JOIN các bảng & tối ưu hoá với Broadcast Join
Nhiệm vụ: kết hợp application với tất cả bảng phụ đã được aggregate.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast

from feature_engineering import (
    aggregate_bureau,
    aggregate_bureau_balance,
    aggregate_previous_application,
    aggregate_installments,
    aggregate_credit_card,
    aggregate_pos_cash,
)


def get_spark(app_name: str = "HomeCreditJoin") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.autoBroadcastJoinThreshold", 50 * 1024 * 1024)  # 50 MB
        .getOrCreate()
    )


def left_join(base: DataFrame, other: DataFrame, on: str) -> DataFrame:
    """Left join với broadcast hint khi bảng phụ nhỏ."""
    return base.join(broadcast(other), on=on, how="left")


def build_final_features(spark: SparkSession, data_dir: str) -> DataFrame:
    """
    Đọc toàn bộ dữ liệu đã làm sạch, tạo features tổng hợp
    và ghép vào bảng application_train.
    """
    app = spark.read.parquet(f"{data_dir}/application_train")
    bureau = spark.read.parquet(f"{data_dir}/bureau")
    bureau_balance = spark.read.parquet(f"{data_dir}/bureau_balance")
    prev = spark.read.parquet(f"{data_dir}/previous_application")
    inst = spark.read.parquet(f"{data_dir}/installments_payments")
    cc = spark.read.parquet(f"{data_dir}/credit_card_balance")
    pos = spark.read.parquet(f"{data_dir}/POS_CASH_balance")

    # Bureau balance → join vào bureau trước
    agg_bb = aggregate_bureau_balance(bureau_balance)
    bureau = bureau.join(agg_bb, on="SK_ID_BUREAU", how="left")

    # Aggregate từng bảng phụ
    agg_bureau = aggregate_bureau(bureau)
    agg_prev = aggregate_previous_application(prev)
    agg_inst = aggregate_installments(inst)
    agg_cc = aggregate_credit_card(cc)
    agg_pos = aggregate_pos_cash(pos)

    # Ghép tuần tự vào application (broadcast join cho các bảng nhỏ)
    result = app
    for agg_df in [agg_bureau, agg_prev, agg_inst, agg_cc, agg_pos]:
        result = left_join(result, agg_df, on="SK_ID_CURR")

    return result


def save_final(df: DataFrame, output_path: str) -> None:
    (
        df.write
        .mode("overwrite")
        .option("compression", "snappy")
        .parquet(output_path)
    )
    print(f"Final features saved → {output_path}  |  rows: {df.count()}")


if __name__ == "__main__":
    spark = get_spark()
    final_df = build_final_features(spark, data_dir="output/cleaned_data")
    save_final(final_df, output_path="output/final_features.parquet")
    spark.stop()
