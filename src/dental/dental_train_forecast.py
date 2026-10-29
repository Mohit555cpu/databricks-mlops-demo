# Databricks notebook source
"""Train and register a dental-claims forecasting candidate for review."""

# MAGIC %run ./dental_forecasting_utils

from datetime import datetime, timezone

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from delta.tables import DeltaTable
from lightgbm import LGBMRegressor
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

dbutils.widgets.text("run_timestamp", "", "Scheduled run timestamp UTC")
dbutils.widgets.text("dataset_snapshot_id", "", "Immutable training data snapshot")
run_timestamp = dbutils.widgets.get("run_timestamp").strip()
dataset_snapshot_id = dbutils.widgets.get("dataset_snapshot_id").strip()
if not dataset_snapshot_id:
    raise ValueError("dataset_snapshot_id is required from the data-generation task")

training_datetime = (
    datetime.fromisoformat(run_timestamp.replace("Z", "+00:00"))
    if run_timestamp
    else datetime.now(timezone.utc)
)
if training_datetime.tzinfo is None:
    training_datetime = training_datetime.replace(tzinfo=timezone.utc)
training_timestamp = training_datetime.astimezone(timezone.utc).isoformat()

snapshot_table = spark.table(SNAPSHOT_TABLE)
snapshot_rows = snapshot_table.filter(
    snapshot_table.dataset_snapshot_id == dataset_snapshot_id
)
snapshot_metadata = snapshot_rows.select("cutoff_month", "source_type").distinct()
metadata = snapshot_metadata.collect()
if len(metadata) != 1:
    raise ValueError(
        f"Expected exactly one cutoff/source type for snapshot "
        f"{dataset_snapshot_id}, found {len(metadata)}"
    )
cutoff_month = pd.Period(metadata[0]["cutoff_month"], freq="M")
source_type = metadata[0]["source_type"]
raw_pdf = snapshot_rows.select(*RAW_COLUMNS).toPandas()
raw_pdf = filter_snapshot_window(raw_pdf, cutoff_month)
monthly_pdf = prepare_monthly_targets(raw_pdf)
supervised_pdf = build_supervised_frame(raw_pdf)
train_pdf, test_pdf, test_start_month = chronological_split(
    supervised_pdf,
    train_fraction=0.8,
)
training_start_month = monthly_pdf["event_month"].min()
training_end_month = monthly_pdf["event_month"].max()
train_target_start = train_pdf["target_month"].min()
train_target_end = train_pdf["target_month"].max()
test_target_start = test_pdf["target_month"].min()
test_target_end = test_pdf["target_month"].max()

training_examples_pdf = pd.concat(
    [
        train_pdf.assign(split="train"),
        test_pdf.assign(split="test"),
    ],
    ignore_index=True,
)
for month_column in ("origin_month", "target_month"):
    training_examples_pdf[month_column] = (
        training_examples_pdf[month_column]
        .dt.to_timestamp()
        .dt.date
    )
training_examples_pdf["dataset_snapshot_id"] = dataset_snapshot_id
training_examples_pdf["cutoff_month"] = str(cutoff_month)
training_examples_pdf["source_type"] = source_type
training_examples_df = spark.createDataFrame(training_examples_pdf)

if not spark.catalog.tableExists(TRAINING_EXAMPLES_TABLE):
    (
        training_examples_df.limit(0)
        .write.format("delta")
        .saveAsTable(TRAINING_EXAMPLES_TABLE)
    )
training_examples_table = DeltaTable.forName(spark, TRAINING_EXAMPLES_TABLE)
(
    training_examples_table.alias("target")
    .merge(
        training_examples_df.alias("source"),
        """
        target.dataset_snapshot_id = source.dataset_snapshot_id
        AND target.split = source.split
        AND target.plan_id = source.plan_id
        AND target.target_month = source.target_month
        AND target.forecast_horizon = source.forecast_horizon
        """,
    )
    .whenNotMatchedInsertAll()
    .execute()
)

X_train = train_pdf[FEATURE_COLUMNS]
y_train = train_pdf[TARGET_COLUMNS]
X_test = test_pdf[FEATURE_COLUMNS]
y_test = test_pdf[TARGET_COLUMNS]

model = MultiOutputRegressor(
    LGBMRegressor(
        objective="regression",
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        random_state=43,
        n_jobs=-1,
        verbosity=-1,
    )
)
model.fit(X_train, y_train)
test_predictions = np.maximum(model.predict(X_test), 0.0)

metric_values: dict[str, float] = {}
for horizon in (1, 2, 3):
    horizon_mask = test_pdf["forecast_horizon"].to_numpy() == horizon
    if not horizon_mask.any():
        raise ValueError(f"No test examples available for horizon {horizon}")
    for target_index, target_name in enumerate(TARGET_COLUMNS):
        metric_name = f"lightgbm_{target_name}_mae_h{horizon}"
        metric_values[metric_name] = float(
            mean_absolute_error(
                y_test.iloc[horizon_mask, target_index],
                test_predictions[horizon_mask, target_index],
            )
        )

# Holt-Winters is a time-series baseline. It is fit only on history available
# at each forecast origin; LightGBM remains the registered candidate.
monthly_pdf = monthly_pdf.sort_values(["plan_id", "event_month"])
test_with_predictions = test_pdf.copy()
test_with_predictions["predicted_frequency"] = test_predictions[:, 0]
test_with_predictions["predicted_cost_pmpm"] = test_predictions[:, 1]
baseline_predictions = {
    target: {1: [], 2: [], 3: []}
    for target in TARGET_COLUMNS
}
baseline_actuals = {
    target: {1: [], 2: [], 3: []}
    for target in TARGET_COLUMNS
}

for (plan_id, origin_month), origin_rows in test_with_predictions.groupby(
    ["plan_id", "origin_month"],
    sort=True,
):
    history = monthly_pdf[
        (monthly_pdf["plan_id"] == plan_id)
        & (monthly_pdf["event_month"] <= origin_month)
    ]
    if len(history) < 24:
        raise ValueError(
            f"Need at least 24 history months for seasonal baseline: {plan_id}"
        )

    for target_name, prediction_column in (
        (TARGET_COLUMNS[0], "predicted_frequency"),
        (TARGET_COLUMNS[1], "predicted_cost_pmpm"),
    ):
        series = history[target_name].to_numpy(dtype=float)
        baseline = ExponentialSmoothing(
            series,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated",
        ).fit(optimized=True)
        forecast = np.maximum(baseline.forecast(3), 0.0)

        for row in origin_rows.itertuples(index=False):
            horizon = int(row.forecast_horizon)
            baseline_predictions[target_name][horizon].append(
                float(forecast[horizon - 1])
            )
            baseline_actuals[target_name][horizon].append(
                float(getattr(row, target_name))
            )

for target_name in TARGET_COLUMNS:
    for horizon in (1, 2, 3):
        metric_values[f"statsmodels_{target_name}_mae_h{horizon}"] = float(
            mean_absolute_error(
                baseline_actuals[target_name][horizon],
                baseline_predictions[target_name][horizon],
            )
        )

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Shared/dental-claims-forecasting")

input_example = X_train.head(5)
signature = infer_signature(input_example, model.predict(input_example))
client = MlflowClient()

with mlflow.start_run(run_name="dental_claims_forecast_candidate") as run:
    mlflow.log_params(
        {
            "algorithm": "MultiOutputRegressor_LightGBM",
            "baseline": "Statsmodels_HoltWinters",
            "forecast_horizon_months": 3,
            "train_fraction": 0.8,
            "dataset_snapshot_id": dataset_snapshot_id,
            "cutoff_month": str(cutoff_month),
            "source_type": source_type,
            "training_start_month": str(training_start_month),
            "training_end_month": str(training_end_month),
            "train_examples": len(train_pdf),
            "test_examples": len(test_pdf),
            "train_target_start_month": str(train_target_start),
            "train_target_end_month": str(train_target_end),
            "test_target_start_month": str(test_target_start),
            "test_target_end_month": str(test_target_end),
            "feature_columns": ",".join(FEATURE_COLUMNS),
            "target_columns": ",".join(TARGET_COLUMNS),
        }
    )
    mlflow.log_metrics(metric_values)
    mlflow.set_tags(
        {
            "model_status": "CANDIDATE_REQUIRES_MANUAL_APPROVAL",
            "training_timestamp_utc": training_timestamp,
            "dataset_snapshot_id": dataset_snapshot_id,
            "cutoff_month": str(cutoff_month),
            "source_type": source_type,
            "training_start_month": str(training_start_month),
            "training_end_month": str(training_end_month),
            "train_target_start_month": str(train_target_start),
            "train_target_end_month": str(train_target_end),
            "test_start_month": str(test_start_month),
            "test_end_month": str(test_target_end),
            "model_name": MODEL_NAME,
        }
    )

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        registered_model_name=MODEL_NAME,
    )

    run_versions = [
        version
        for version in client.search_model_versions(f"name='{MODEL_NAME}'")
        if version.run_id == run.info.run_id and version.status == "READY"
    ]
    if not run_versions:
        raise RuntimeError(
            f"Model artifact was logged, but no READY registered version "
            f"was found for run {run.info.run_id} in {MODEL_NAME}"
        )
    candidate_version = max(run_versions, key=lambda version: int(version.version))
    model_version_tags = {
        "approval_status": "PENDING_REVIEW",
        "dataset_snapshot_id": dataset_snapshot_id,
        "cutoff_month": str(cutoff_month),
        "source_type": source_type,
        "training_start_month": str(training_start_month),
        "training_end_month": str(training_end_month),
        "train_target_start_month": str(train_target_start),
        "train_target_end_month": str(train_target_end),
        "test_target_start_month": str(test_target_start),
        "test_target_end_month": str(test_target_end),
        "test_start_month": str(test_start_month),
    }
    for tag_key, tag_value in model_version_tags.items():
        client.set_model_version_tag(
            MODEL_NAME,
            candidate_version.version,
            tag_key,
            tag_value,
        )

    print(f"Candidate model: {MODEL_NAME}")
    print(f"Candidate version: {candidate_version.version}")
    print(f"Candidate URI: {model_info.model_uri}")
    print(f"Dataset snapshot ID: {dataset_snapshot_id}")
    print(f"Training window: {training_start_month} through {training_end_month}")
    print(f"Train target months: {train_target_start} through {train_target_end}")
    print(f"Test target months: {test_target_start} through {test_target_end}")
    print(f"Saved split data: {TRAINING_EXAMPLES_TABLE}")
    print(f"Training timestamp UTC: {training_timestamp}")
    print("Candidate is registered but NOT approved or promoted.")
    print(f"LightGBM and Statsmodels comparison metrics: {metric_values}")
