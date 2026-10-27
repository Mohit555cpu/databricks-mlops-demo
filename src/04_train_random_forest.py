
# Databricks notebook source
"""Train, evaluate, trace and conditionally register a Random Forest model."""

import os
from datetime import datetime, timezone

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

# ---- Configuration ----
CATALOG = "my_feature_store"
SCHEMA = "retail_ml"

TRAIN_TABLE = f"{CATALOG}.{SCHEMA}.train_dataset"
TEST_TABLE = f"{CATALOG}.{SCHEMA}.test_dataset"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.random_forest_customer_rating"
EXPERIMENT_PATH = "/Shared/retail-mlops-practice"

FEATURES = ["total_purchases_30d"]
TARGET = "rating"

MAX_MAE = float(os.getenv("MAX_MAE", "0.25"))

MODEL_PARAMS = {
    "n_estimators": 150,
    "max_depth": 8,
    "min_samples_leaf": 3,
    "random_state": 43,
    "n_jobs": -1,
}

# ---- Load and validate data ----
train_pdf = spark.table(TRAIN_TABLE).toPandas()
test_pdf = spark.table(TEST_TABLE).toPandas()

if train_pdf.empty or test_pdf.empty:
    raise ValueError("Training and test datasets must both contain rows")

required_columns = set(FEATURES + [TARGET])
missing_train = required_columns - set(train_pdf.columns)
missing_test = required_columns - set(test_pdf.columns)

if missing_train or missing_test:
    raise ValueError(
        f"Missing columns: train={sorted(missing_train)}, "
        f"test={sorted(missing_test)}"
    )

X_train = train_pdf[FEATURES]
y_train = train_pdf[TARGET]
X_test = test_pdf[FEATURES]
y_test = test_pdf[TARGET]

if y_test.isna().any():
    raise ValueError("Test target contains null values")

if (y_test == 0).any():
    raise ValueError(
        "MAPE is undefined for zero target values. "
        "Choose a suitable metric or handle zero values explicitly."
    )

print(f"Training rows: {len(train_pdf)}")
print(f"Test rows: {len(test_pdf)}")

# ---- Train model ----
model = RandomForestRegressor(**MODEL_PARAMS)
model.fit(X_train, y_train)

# ---- Evaluate ----
predictions = model.predict(X_test)

mae = float(mean_absolute_error(y_test, predictions))
rmse = float(mean_squared_error(y_test, predictions) ** 0.5)
r2 = float(r2_score(y_test, predictions))
mape = float(mean_absolute_percentage_error(y_test, predictions) * 100)

print(f"MAE: {mae:.4f} (required <= {MAX_MAE:.4f})")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")
print(f"MAPE: {mape:.4f}%")

# ---- Configure MLflow ----
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)

input_example = X_train.head(5)
signature = infer_signature(input_example, model.predict(input_example))

# ---- Traced prediction function ----
# Only a small sample is traced to avoid capturing the full test dataset.
@mlflow.trace
def traced_predict(input_data: pd.DataFrame) -> list:
    """Capture a prediction call as an MLflow trace."""
    result = model.predict(input_data)
    return result.tolist()


# ---- Log run, trace, metrics and register model ----
training_timestamp = datetime.now(timezone.utc).isoformat()

with mlflow.start_run(run_name="random_forest_candidate") as run:
    mlflow.log_params(MODEL_PARAMS)

    mlflow.log_params({
        "feature_columns": ",".join(FEATURES),
        "target_column": TARGET,
        "max_mae_threshold": MAX_MAE,
        "train_rows": len(train_pdf),
        "test_rows": len(test_pdf),
    })

    mlflow.log_param("training_timestamp_utc", training_timestamp)

    mlflow.log_metrics({
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape_percent": mape,
    })

    # This call creates an MLflow trace for a small prediction sample.
    sample_input = X_test.head(5)
    sample_predictions = traced_predict(sample_input)

    print(f"Trace sample predictions: {sample_predictions}")
    print(f"MLflow Run ID: {run.info.run_id}")
    print(f"Training timestamp UTC: {training_timestamp}")

    # Quality gate: do not register a model that fails the threshold.
    if mae > MAX_MAE:
        mlflow.set_tag("quality_gate", "FAILED")
        raise ValueError(
            f"Quality gate failed: MAE {mae:.4f} exceeds "
            f"threshold {MAX_MAE:.4f}. Model was not registered."
        )

    mlflow.set_tag("quality_gate", "PASSED")
    mlflow.set_tag("model_family", "RandomForestRegressor")

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        registered_model_name=MODEL_NAME,
    )

    print(f"Quality gate: PASSED")
    print(f"Model URI: {model_info.model_uri}")
    print(f"Registered model: {MODEL_NAME}")

print("Training, evaluation, tracing and registration completed.")
