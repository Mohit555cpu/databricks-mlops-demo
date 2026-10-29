# Databricks notebook source
"""Smoke-test retail model inference against a small input sample."""

from uuid import uuid4

import mlflow
from pyspark.sql import functions as F

MODEL_NAME = "my_feature_store.retail_ml.random_forest_customer_rating"
INPUT_TABLE = "retail_input.monthly.customer_features"
STAGING_TABLE = (
    "retail_output.predictions.customer_rating_predictions_smoke_test"
)
FEATURE_COLUMN = "total_purchases_30d"
SAMPLE_SIZE = 10

dbutils.widgets.text("model_version", "7", "Registered model version")
dbutils.widgets.text("batch_month", "", "Input batch month (YYYY-MM-DD)")

MODEL_VERSION = dbutils.widgets.get("model_version").strip()
BATCH_MONTH = dbutils.widgets.get("batch_month").strip()
BATCH_ID = f"smoke-{uuid4()}"

if not MODEL_VERSION.isdigit() or int(MODEL_VERSION) < 1:
    raise ValueError("model_version must be a positive registered model version")

mlflow.set_registry_uri("databricks-uc")
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
predict = mlflow.pyfunc.spark_udf(spark, model_uri, result_type="double")
print(f"Loaded registered model: {model_uri}")

input_table = spark.table(INPUT_TABLE)
input_columns = {field.name for field in input_table.schema.fields}
required_input_columns = {"customer_id", FEATURE_COLUMN, "batch_month"}
missing_columns = required_input_columns - input_columns
if missing_columns:
    raise ValueError(f"Input table is missing columns: {sorted(missing_columns)}")

if not BATCH_MONTH:
    latest_batch_month = input_table.agg(
        F.max("batch_month").alias("latest_batch_month")
    ).first()["latest_batch_month"]
    if latest_batch_month is None:
        raise ValueError(f"No input batches found in {INPUT_TABLE}")
    BATCH_MONTH = latest_batch_month.isoformat()

sample_input = (
    input_table
    .filter(F.col("batch_month") == F.to_date(F.lit(BATCH_MONTH)))
    .select("customer_id", FEATURE_COLUMN, "batch_month")
    .orderBy("customer_id")
    .limit(SAMPLE_SIZE)
)
sample_count = sample_input.count()
if sample_count == 0:
    raise ValueError(f"No input rows found for batch_month={BATCH_MONTH}")

invalid_input = sample_input.filter(
    F.col("customer_id").isNull()
    | F.col(FEATURE_COLUMN).isNull()
    | F.col("batch_month").isNull()
)
if invalid_input.limit(1).count():
    raise ValueError("Smoke-test input contains null required values")

duplicate_ids = (
    sample_input.groupBy("customer_id")
    .count()
    .filter(F.col("count") > 1)
)
if duplicate_ids.limit(1).count():
    raise ValueError("Smoke-test input contains duplicate customer_id values")

smoke_output = (
    sample_input
    .withColumn("predicted_rating", predict(F.col(FEATURE_COLUMN)))
    .withColumn("model_version", F.lit(MODEL_VERSION))
    .withColumn("batch_id", F.lit(BATCH_ID))
    .withColumn("model_name", F.lit(MODEL_NAME))
    .withColumn("prediction_timestamp", F.current_timestamp())
    .select(
        "customer_id",
        "predicted_rating",
        "model_version",
        "batch_id",
        "model_name",
        "batch_month",
        "prediction_timestamp",
    )
)

invalid_predictions = smoke_output.filter(
    F.col("predicted_rating").isNull()
    | F.isnan("predicted_rating")
)
if invalid_predictions.limit(1).count():
    raise ValueError("Model produced a null or NaN prediction")

spark.sql(
    f"""
    CREATE TABLE IF NOT EXISTS {STAGING_TABLE} (
      customer_id STRING,
      predicted_rating DOUBLE,
      model_version STRING,
      batch_id STRING,
      model_name STRING,
      batch_month DATE,
      prediction_timestamp TIMESTAMP
    )
    USING DELTA
    """
)

required_output_columns = {
    "customer_id",
    "predicted_rating",
    "model_version",
    "batch_id",
}
output_columns = {field.name for field in spark.table(STAGING_TABLE).schema.fields}
missing_output_columns = required_output_columns - output_columns
if missing_output_columns:
    raise ValueError(
        f"Staging table is missing columns: {sorted(missing_output_columns)}"
    )

smoke_output.write.format("delta").mode("append").saveAsTable(STAGING_TABLE)

saved_batch = spark.table(STAGING_TABLE).filter(F.col("batch_id") == BATCH_ID)
saved_count = saved_batch.count()
if saved_count != sample_count:
    raise ValueError(
        f"Expected {sample_count} saved predictions, found {saved_count}"
    )

print(f"Input table: {INPUT_TABLE}")
print(f"Input batch month: {BATCH_MONTH}")
print(f"Input/prediction rows: {sample_count}")
print(f"Staging output table: {STAGING_TABLE}")
print(f"Verified smoke-test batch: {BATCH_ID}")
