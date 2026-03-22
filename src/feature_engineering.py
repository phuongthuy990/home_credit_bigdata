"""
Thành viên 2 – Feature Engineering
Nhiệm vụ: GROUP BY, aggregations trên các bảng phụ.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F


def get_spark(app_name: str = "HomeCreditFeatures") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


# ── Bureau aggregations ────────────────────────────────────────────────────────

def aggregate_bureau(bureau: DataFrame) -> DataFrame:
    """Tổng hợp thông tin tín dụng lịch sử từ bảng bureau."""
    return bureau.groupBy("SK_ID_CURR").agg(
        F.count("SK_ID_BUREAU").alias("bureau_loan_count"),
        F.sum("AMT_CREDIT_SUM").alias("bureau_total_credit"),
        F.sum("AMT_CREDIT_SUM_DEBT").alias("bureau_total_debt"),
        F.mean("DAYS_CREDIT").alias("bureau_avg_days_credit"),
        F.sum(F.when(F.col("CREDIT_ACTIVE") == "Active", 1).otherwise(0))
         .alias("bureau_active_loans"),
    )


def aggregate_bureau_balance(bureau_balance: DataFrame) -> DataFrame:
    """Tổng hợp trạng thái hàng tháng từ bureau_balance."""
    return bureau_balance.groupBy("SK_ID_BUREAU").agg(
        F.count("MONTHS_BALANCE").alias("bb_months_count"),
        F.mean("STATUS").alias("bb_avg_status"),
    )


# ── Previous application aggregations ─────────────────────────────────────────

def aggregate_previous_application(prev: DataFrame) -> DataFrame:
    return prev.groupBy("SK_ID_CURR").agg(
        F.count("SK_ID_PREV").alias("prev_app_count"),
        F.mean("AMT_APPLICATION").alias("prev_avg_application"),
        F.mean("AMT_CREDIT").alias("prev_avg_credit"),
        F.sum(F.when(F.col("NAME_CONTRACT_STATUS") == "Approved", 1).otherwise(0))
         .alias("prev_approved_count"),
        F.sum(F.when(F.col("NAME_CONTRACT_STATUS") == "Refused", 1).otherwise(0))
         .alias("prev_refused_count"),
    )


# ── Installments aggregations ──────────────────────────────────────────────────

def aggregate_installments(inst: DataFrame) -> DataFrame:
    inst = inst.withColumn(
        "payment_diff",
        F.col("AMT_INSTALMENT") - F.col("AMT_PAYMENT")
    ).withColumn(
        "days_late",
        F.col("DAYS_ENTRY_PAYMENT") - F.col("DAYS_INSTALMENT")
    )
    return inst.groupBy("SK_ID_CURR").agg(
        F.count("SK_ID_PREV").alias("inst_count"),
        F.mean("payment_diff").alias("inst_avg_payment_diff"),
        F.mean("days_late").alias("inst_avg_days_late"),
        F.sum(F.when(F.col("days_late") > 0, 1).otherwise(0)).alias("inst_late_count"),
    )


# ── Credit card aggregations ───────────────────────────────────────────────────

def aggregate_credit_card(cc: DataFrame) -> DataFrame:
    return cc.groupBy("SK_ID_CURR").agg(
        F.mean("AMT_BALANCE").alias("cc_avg_balance"),
        F.mean("AMT_CREDIT_LIMIT_ACTUAL").alias("cc_avg_credit_limit"),
        F.mean("CNT_DRAWINGS_TOTAL").alias("cc_avg_drawings"),
    )


# ── POS CASH aggregations ──────────────────────────────────────────────────────

def aggregate_pos_cash(pos: DataFrame) -> DataFrame:
    return pos.groupBy("SK_ID_CURR").agg(
        F.count("SK_ID_PREV").alias("pos_count"),
        F.mean("CNT_INSTALMENT").alias("pos_avg_instalment"),
        F.sum(F.when(F.col("NAME_CONTRACT_STATUS") == "Active", 1).otherwise(0))
         .alias("pos_active_count"),
    )


if __name__ == "__main__":
    spark = get_spark()
    bureau = spark.read.parquet("output/cleaned_data/bureau")
    agg_bureau = aggregate_bureau(bureau)
    agg_bureau.show(5)
    spark.stop()
