# Databricks notebook source
"""Score a retail customer batch with an explicitly selected UC model version."""

import mlflow
from delta.tables import DeltaTable
from pyspark.sql import functions as F

MODEL_NAME = "my_feature_store.retail_ml.random_forest_customer_rating"
INPUT_TABLE = "retail_input.monthly.customer_features"
OUTPUT_TABLE = "retail_output.predictions.customer_rating_predictions"
FEATURE_COLUMN = "total_purchases_30d"

dbutils.widgets.text("model_version", "7", "Approved model version")
dbutils.widgets.text("batch_month", "", "Input batch month (YYYY-MM-DD)")
dbutils.widgets.text("batch_id", "", "Unique batch ID")

MODEL_VERSION = dbutils.widgets.get("model_version").strip()
BATCH_MONTH = dbutils.widgets.get("batch_month").strip()
BATCH_ID = dbutils.widgets.get("batch_id").strip()

if not MODEL_VERSION.isdigit() or int(MODEL_VERSION) < 1:
    raise ValueError("model_version must be a positive registered model version")

if not BATCH_ID:
    raise ValueError("batch_id is required")

mlflow.set_registry_uri("databricks-uc")
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
predict = mlflow.pyfunc.spark_udf(
    spark,
    model_uri,
    result_type="double",
)

input_schema = spark.table(INPUT_TABLE).schema
input_columns = {field.name for field in input_schema.fields}
required_columns = {"customer_id", FEATURE_COLUMN, "batch_month"}
missing_columns = required_columns - input_columns
if missing_columns:
    raise ValueError(f"Input table is missing columns: {sorted(missing_columns)}")

input_table = spark.table(INPUT_TABLE)
if not BATCH_MONTH:
    latest_batch_month = input_table.agg(
        F.max("batch_month").alias("latest_batch_month")
    ).first()["latest_batch_month"]
    if latest_batch_month is None:
        raise ValueError(f"No input batches found in {INPUT_TABLE}")
    BATCH_MONTH = latest_batch_month.isoformat()

batch_input = input_table.filter(
    F.col("batch_month") == F.to_date(F.lit(BATCH_MONTH))
).select("customer_id", FEATURE_COLUMN, "batch_month")

if batch_input.limit(1).count() == 0:
    raise ValueError(f"No input rows found for batch_month={BATCH_MONTH}")

invalid_rows = batch_input.filter(
    F.col("customer_id").isNull()
    | F.col(FEATURE_COLUMN).isNull()
    | F.col("batch_month").isNull()
)
if invalid_rows.limit(1).count() > 0:
    raise ValueError("Input batch contains null IDs, features, or batch_month values")

duplicate_customers = (
    batch_input.groupBy("customer_id")
    .count()
    .filter(F.col("count") > 1)
)
if duplicate_customers.limit(1).count() > 0:
    raise ValueError("Input batch contains duplicate customer_id values")

output_batch = (
    batch_input
    .withColumn(
        "predicted_rating",
        predict(F.col(FEATURE_COLUMN)),
    )
    .withColumn("model_name", F.lit(MODEL_NAME))
    .withColumn("model_version", F.lit(MODEL_VERSION))
    .withColumn("batch_id", F.lit(BATCH_ID))
    .withColumn("prediction_timestamp", F.current_timestamp())
    .select(
        "customer_id",
        "predicted_rating",
        "model_name",
        "model_version",
        "batch_id",
        "batch_month",
        "prediction_timestamp",
    )
)

invalid_predictions = output_batch.filter(F.col("predicted_rating").isNull())
if invalid_predictions.limit(1).count() > 0:
    raise ValueError("Model produced null predictions")

spark.sql(
    f"""
    CREATE TABLE IF NOT EXISTS {OUTPUT_TABLE} (
      customer_id STRING,
      predicted_rating DOUBLE,
      model_name STRING,
      model_version STRING,
      batch_id STRING,
      batch_month DATE,
      prediction_timestamp TIMESTAMP
    )
    USING DELTA
    """
)

target = DeltaTable.forName(spark, OUTPUT_TABLE)
(
    target.alias("target")
    .merge(
        output_batch.alias("source"),
        """
        target.batch_id = source.batch_id
        AND target.customer_id = source.customer_id
        """,
    )
    .whenMatchedUpdate(
        set={
            "predicted_rating": "source.predicted_rating",
            "model_name": "source.model_name",
            "model_version": "source.model_version",
            "batch_month": "source.batch_month",
            "prediction_timestamp": "source.prediction_timestamp",
        }
    )
    .whenNotMatchedInsertAll()
    .execute()
)

saved_rows = (
    spark.table(OUTPUT_TABLE)
    .filter(F.col("batch_id") == BATCH_ID)
    .count()
)

print(f"Model: {model_uri}")
print(f"Input table: {INPUT_TABLE}")
print(f"Output table: {OUTPUT_TABLE}")
print(f"Batch ID: {BATCH_ID}")
print(f"Batch month: {BATCH_MONTH}")
print(f"Saved prediction rows: {saved_rows}")
