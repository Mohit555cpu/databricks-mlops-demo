# Databricks notebook source
"""Train, evaluate, and conditionally register a Random Forest model.

Run on Databricks compute with access to the Unity Catalog tables/model.
Set MAX_MAE as a Job environment variable after confirming the team's threshold.
"""

import os
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---- Configuration: keep shared values in one place ----
CATALOG = "my_feature_store"
SCHEMA = "retail_ml"
TRAIN_TABLE = f"{CATALOG}.{SCHEMA}.train_dataset"
TEST_TABLE = f"{CATALOG}.{SCHEMA}.test_dataset"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.random_forest_customer_rating"
EXPERIMENT_PATH = "/Shared/retail-mlops-practice"
FEATURES = ["total_purchases_30d"]
TARGET = "rating"

# Example default only: confirm this acceptance threshold with the Data Scientist.
MAX_MAE = float(os.getenv("MAX_MAE", "0.25"))

# One candidate configuration per run; change parameters here when testing a new version.
MODEL_PARAMS = {
    "n_estimators": 150,
    "max_depth": 8,
    "min_samples_leaf": 3,
    "random_state": 43,
    "n_jobs": -1,
}

# ---- Load data once ----
train_df = spark.table(TRAIN_TABLE)
test_df = spark.table(TEST_TABLE)
train_pdf = train_df.toPandas()
test_pdf = test_df.toPandas()

if train_pdf.empty or test_pdf.empty:
    raise ValueError("Training and test datasets must both contain rows")

missing_train = set(FEATURES + [TARGET]) - set(train_pdf.columns)
missing_test = set(FEATURES + [TARGET]) - set(test_pdf.columns)
if missing_train or missing_test:
    raise ValueError(
        f"Missing columns: train={sorted(missing_train)}, test={sorted(missing_test)}"
    )

X_train, y_train = train_pdf[FEATURES], train_pdf[TARGET]
X_test, y_test = test_pdf[FEATURES], test_pdf[TARGET]
print(f"Training rows: {len(train_pdf)} | Test rows: {len(test_pdf)}")

# ---- Train and evaluate one candidate ----
model = RandomForestRegressor(**MODEL_PARAMS)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mae = float(mean_absolute_error(y_test, predictions))
rmse = float(mean_squared_error(y_test, predictions) ** 0.5)
r2 = float(r2_score(y_test, predictions))

print(f"MAE: {mae:.4f} (required <= {MAX_MAE:.4f})")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

# ---- Track every evaluation; register only if the quality gate passes ----
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)
input_example = X_train.head(5)
signature = infer_signature(input_example, model.predict(input_example))

with mlflow.start_run(run_name="random_forest_candidate") as run:
    mlflow.log_params(MODEL_PARAMS)
    mlflow.log_param("feature_columns", ",".join(FEATURES))
    mlflow.log_param("target_column", TARGET)
    mlflow.log_param("max_mae_threshold", MAX_MAE)
    mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})

    if mae > MAX_MAE:
        mlflow.set_tag("quality_gate", "FAILED")
        raise ValueError(
            f"Quality gate failed: MAE {mae:.4f} exceeds threshold {MAX_MAE:.4f}. "
            "Model was not registered."
        )

    mlflow.set_tag("quality_gate", "PASSED")
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        registered_model_name=MODEL_NAME,
    )
    print(f"Quality gate passed. Run ID: {run.info.run_id}")
    print(f"Model URI: {model_info.model_uri}")
    print(f"Registered model: {MODEL_NAME}")

print("Training/evaluation/conditional registration completed successfully.")
