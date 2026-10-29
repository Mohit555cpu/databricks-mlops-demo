# Databricks notebook source
"""Generate deterministic synthetic monthly plan-level dental claims data."""

# MAGIC %run ./dental_forecasting_utils

from delta.tables import DeltaTable
from pyspark.sql import functions as F

dbutils.widgets.text("run_timestamp", "", "Scheduled run timestamp")
run_timestamp = dbutils.widgets.get("run_timestamp").strip()
cutoff_month = resolve_cutoff_month(
    reference_timestamp=run_timestamp or None,
)
snapshot_id = snapshot_id_for_cutoff(cutoff_month)

spark.sql("CREATE SCHEMA IF NOT EXISTS dental_forecasting.source")
spark.sql("CREATE SCHEMA IF NOT EXISTS dental_forecasting.training_data")
spark.sql("CREATE SCHEMA IF NOT EXISTS dental_forecasting.models")
spark.sql("CREATE SCHEMA IF NOT EXISTS dental_forecasting.predictions")

generated_pdf = generate_synthetic_claims(
    months=SNAPSHOT_MONTHS,
    seed=43,
    end_month=cutoff_month,
)
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

cutoff_start = cutoff_month.start_time.date()
snapshot_pdf = filter_snapshot_window(
    spark.table(SOURCE_TABLE)
    .filter(F.col("event_month") >= F.lit(cutoff_start))
    .toPandas(),
    cutoff_month,
)
prepare_monthly_targets(snapshot_pdf)
snapshot_df = (
    spark.createDataFrame(snapshot_pdf)
    .withColumn("dataset_snapshot_id", F.lit(snapshot_id))
    .withColumn("cutoff_month", F.lit(str(cutoff_month)))
    .withColumn("source_type", F.lit("synthetic_demo"))
    .withColumn("snapshot_created_at", F.current_timestamp())
)

if not spark.catalog.tableExists(SNAPSHOT_TABLE):
    snapshot_df.limit(0).write.format("delta").saveAsTable(SNAPSHOT_TABLE)

snapshot_table = DeltaTable.forName(spark, SNAPSHOT_TABLE)
(
    snapshot_table.alias("target")
    .merge(
        snapshot_df.alias("source"),
        """
        target.dataset_snapshot_id = source.dataset_snapshot_id
        AND target.plan_id = source.plan_id
        AND target.event_month = source.event_month
        """,
    )
    .whenNotMatchedInsertAll()
    .execute()
)

snapshot_rows = (
    spark.table(SNAPSHOT_TABLE)
    .filter(F.col("dataset_snapshot_id") == snapshot_id)
    .count()
)
expected_rows = len(generated_pdf)
if snapshot_rows != expected_rows:
    raise ValueError(
        f"Expected {expected_rows} snapshot rows for {snapshot_id}, "
        f"found {snapshot_rows}"
    )

dbutils.jobs.taskValues.set(key="dataset_snapshot_id", value=snapshot_id)
print(f"Source table: {SOURCE_TABLE}")
print(f"Dataset snapshot ID: {snapshot_id}")
print(f"Cutoff month: {cutoff_month}")
print(
    f"Snapshot window: {cutoff_month - (SNAPSHOT_MONTHS - 1)} "
    f"through {cutoff_month}"
)
print(f"Snapshot rows verified: {snapshot_rows}")
print("Source type: synthetic_demo; not genuine claims and not for business decisions.")
display(
    spark.table(SNAPSHOT_TABLE)
    .filter(F.col("dataset_snapshot_id") == snapshot_id)
    .groupBy("dataset_snapshot_id", "dataset_version")
    .count()
    .orderBy("dataset_version")
)
