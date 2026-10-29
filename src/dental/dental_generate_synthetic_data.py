# Databricks notebook source
"""Generate deterministic synthetic monthly plan-level dental claims data."""

# MAGIC %run ./dental_forecasting_utils

from delta.tables import DeltaTable

spark.sql("CREATE SCHEMA IF NOT EXISTS dental_forecasting.source")
spark.sql("CREATE SCHEMA IF NOT EXISTS dental_forecasting.models")
spark.sql("CREATE SCHEMA IF NOT EXISTS dental_forecasting.predictions")

generated_pdf = generate_synthetic_claims(months=120, seed=43)
generated_df = spark.createDataFrame(generated_pdf)

if not spark.catalog.tableExists(SOURCE_TABLE):
    (
        generated_df.limit(0)
        .write.format("delta")
        .saveAsTable(SOURCE_TABLE)
    )

target = DeltaTable.forName(spark, SOURCE_TABLE)
(
    target.alias("target")
    .merge(
        generated_df.alias("source"),
        """
        target.plan_id = source.plan_id
        AND target.event_month = source.event_month
        """,
    )
    .whenMatchedUpdateAll()
    .whenNotMatchedInsertAll()
    .execute()
)

print(f"Source table: {SOURCE_TABLE}")
print(f"Synthetic rows generated: {len(generated_pdf)}")
print("Data is synthetic and for demo/testing only, not for business decisions.")
display(
    spark.table(SOURCE_TABLE)
    .groupBy("dataset_version")
    .count()
    .orderBy("dataset_version")
)
