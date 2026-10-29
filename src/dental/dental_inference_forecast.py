# Databricks notebook source
"""Forecast three months with the manually approved dental Champion model."""

# MAGIC %run ./dental_forecasting_utils

from datetime import datetime, timezone
from uuid import uuid4

import mlflow
import numpy as np
from delta.tables import DeltaTable
from mlflow import MlflowClient
from pyspark.sql import functions as F

dbutils.widgets.text("batch_id", "", "Forecast run ID (optional)")
batch_id = dbutils.widgets.get("batch_id").strip() or str(uuid4())

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
champion = client.get_model_version_by_alias(MODEL_NAME, "Champion")
model_uri = f"models:/{MODEL_NAME}@Champion"
model = mlflow.pyfunc.load_model(model_uri)

raw_pdf = spark.table(SOURCE_TABLE).toPandas()
forecast_rows, forecast_features = build_future_features(
    raw_pdf,
    horizon_months=3,
)
predictions = np.maximum(model.predict(forecast_features), 0.0)
if predictions.shape != (len(forecast_rows), len(TARGET_COLUMNS)):
    raise ValueError(
        "Champion model returned an unexpected output shape: "
        f"{predictions.shape}"
    )
if not np.isfinite(predictions).all():
    raise ValueError("Champion model returned a null or non-finite forecast")

forecast_rows["predicted_claims_frequency_pmpm"] = predictions[:, 0]
forecast_rows["predicted_claims_cost_pmpm"] = predictions[:, 1]
forecast_rows["model_name"] = MODEL_NAME
forecast_rows["model_version"] = str(champion.version)
forecast_rows["batch_id"] = batch_id
forecast_rows["forecast_run_timestamp"] = datetime.now(timezone.utc).replace(
    tzinfo=None
)
forecast_rows["origin_month"] = (
    forecast_rows["origin_month"].dt.to_timestamp().dt.date
)
forecast_rows["forecast_month"] = (
    forecast_rows["forecast_month"].dt.to_timestamp().dt.date
)

output_pdf = forecast_rows[
    [
        "plan_id",
        "origin_month",
        "forecast_month",
        "forecast_horizon",
        "predicted_claims_frequency_pmpm",
        "predicted_claims_cost_pmpm",
        "model_name",
        "model_version",
        "batch_id",
        "forecast_run_timestamp",
    ]
]
output_df = spark.createDataFrame(output_pdf)

spark.sql("CREATE SCHEMA IF NOT EXISTS dental_forecasting.predictions")
spark.sql(
    f"""
    CREATE TABLE IF NOT EXISTS {FORECAST_TABLE} (
      plan_id STRING,
      origin_month DATE,
      forecast_month DATE,
      forecast_horizon INT,
      predicted_claims_frequency_pmpm DOUBLE,
      predicted_claims_cost_pmpm DOUBLE,
      model_name STRING,
      model_version STRING,
      batch_id STRING,
      forecast_run_timestamp TIMESTAMP
    )
    USING DELTA
    """
)

target = DeltaTable.forName(spark, FORECAST_TABLE)
(
    target.alias("target")
    .merge(
        output_df.alias("source"),
        """
        target.batch_id = source.batch_id
        AND target.plan_id = source.plan_id
        AND target.forecast_month = source.forecast_month
        """,
    )
    .whenMatchedUpdateAll()
    .whenNotMatchedInsertAll()
    .execute()
)

saved_count = (
    spark.table(FORECAST_TABLE)
    .filter(F.col("batch_id") == batch_id)
    .count()
)
if saved_count != len(output_pdf):
    raise ValueError(
        f"Expected {len(output_pdf)} forecasts in output, found {saved_count}"
    )

print(f"Approved model: {MODEL_NAME}, version {champion.version} (Champion)")
print(f"Forecast output: {FORECAST_TABLE}")
print(f"Forecast batch ID: {batch_id}")
print(f"Saved forecast rows: {saved_count}")
display(
    spark.table(FORECAST_TABLE)
    .filter(F.col("batch_id") == batch_id)
    .orderBy("plan_id", "forecast_month")
)
