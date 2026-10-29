# Databricks notebook source
"""Smoke-test retail model fitting without reading/registering production data."""

import math

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

features = pd.DataFrame(
    {"total_purchases_30d": [1, 2, 4, 8, 12, 16, 20, 24]}
)
target = pd.Series([1.1, 1.4, 1.8, 2.5, 3.0, 3.7, 4.2, 4.8])

model = RandomForestRegressor(
    n_estimators=5,
    max_depth=3,
    random_state=43,
    n_jobs=1,
)
model.fit(features, target)

predictions = model.predict(features.head(3))
if len(predictions) != 3:
    raise ValueError(f"Expected 3 predictions, got {len(predictions)}")
if not all(math.isfinite(float(value)) for value in predictions):
    raise ValueError("Training smoke test produced a null or non-finite prediction")

print("Training smoke test passed")
print(f"Fitted model: {type(model).__name__}")
print(f"Predictions produced: {len(predictions)}")
print("No production data was read and no model was registered.")
