# Databricks notebook source
"""Train and register a dental-claims forecasting candidate for review."""

# MAGIC %run ./dental_forecasting_utils

from datetime import datetime, timezone

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

dbutils.widgets.text("run_timestamp", "", "Scheduled run timestamp UTC")
run_timestamp = dbutils.widgets.get("run_timestamp").strip()
training_datetime = (
    datetime.fromisoformat(run_timestamp.replace("Z", "+00:00"))
    if run_timestamp
    else datetime.now(timezone.utc)
)
if training_datetime.tzinfo is None:
    training_datetime = training_datetime.replace(tzinfo=timezone.utc)
training_timestamp = training_datetime.astimezone(timezone.utc).isoformat()

raw_pdf = spark.table(SOURCE_TABLE).toPandas()
monthly_pdf = prepare_monthly_targets(raw_pdf)
supervised_pdf = build_supervised_frame(raw_pdf)
train_pdf, test_pdf, test_start_month = chronological_split(
    supervised_pdf,
    train_fraction=0.8,
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

latest_dataset_month = monthly_pdf["event_month"].max()
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
            "train_examples": len(train_pdf),
            "test_examples": len(test_pdf),
            "test_start_month": str(test_start_month),
            "latest_dataset_month": str(latest_dataset_month),
            "feature_columns": ",".join(FEATURE_COLUMNS),
            "target_columns": ",".join(TARGET_COLUMNS),
        }
    )
    mlflow.log_metrics(metric_values)
    mlflow.set_tags(
        {
            "model_status": "CANDIDATE_REQUIRES_MANUAL_APPROVAL",
            "training_timestamp_utc": training_timestamp,
            "dataset_version": str(latest_dataset_month),
            "test_start_month": str(test_start_month),
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
    client.set_model_version_tag(
        MODEL_NAME,
        candidate_version.version,
        "approval_status",
        "PENDING_REVIEW",
    )

    print(f"Candidate model: {MODEL_NAME}")
    print(f"Candidate version: {candidate_version.version}")
    print(f"Candidate URI: {model_info.model_uri}")
    print(f"Latest dataset month: {latest_dataset_month}")
    print(f"Chronological test starts: {test_start_month}")
    print(f"Training timestamp UTC: {training_timestamp}")
    print("Candidate is registered but NOT approved or promoted.")
    print(f"LightGBM and Statsmodels comparison metrics: {metric_values}")
