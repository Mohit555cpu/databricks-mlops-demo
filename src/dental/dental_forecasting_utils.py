# Databricks notebook source
# Databricks notebook source
"""Synthetic dental claims data and time-aware forecasting utilities."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

SOURCE_TABLE = "dental_forecasting.source.plan_monthly_claims"
MODEL_NAME = "dental_forecasting.models.dental_claims_forecast"
FORECAST_TABLE = "dental_forecasting.predictions.plan_claims_forecast"

FEATURE_COLUMNS = [
    "plan_index",
    "forecast_horizon",
    "target_month_sin",
    "target_month_cos",
    "frequency_lag_1",
    "frequency_lag_2",
    "frequency_lag_3",
    "frequency_lag_12",
    "cost_pmpm_lag_1",
    "cost_pmpm_lag_2",
    "cost_pmpm_lag_3",
    "cost_pmpm_lag_12",
]
TARGET_COLUMNS = ["claims_frequency_pmpm", "claims_cost_pmpm"]
RAW_COLUMNS = [
    "plan_id",
    "event_month",
    "member_months",
    "claim_count",
    "total_claim_amount",
    "dataset_version",
]


def _has_monthly_gaps(months: pd.Series) -> bool:
    ordinals = months.array.asi8
    return bool((np.diff(ordinals) != 1).any())


def _stable_seed(seed: int, plan_id: str, month: pd.Period) -> int:
    key = f"{seed}:{plan_id}:{month}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def generate_synthetic_claims(
    months: int = 120,
    seed: int = 43,
    end_month: str | pd.Period | None = None,
) -> pd.DataFrame:
    """Create reproducible monthly plan-level claims and exposure data."""
    if months < 36:
        raise ValueError("At least 36 months are required for seasonal forecasting")

    if end_month is None:
        current_month = pd.Timestamp.now(tz="UTC").tz_localize(None).to_period("M")
        final_month = current_month - 1
    else:
        final_month = pd.Period(end_month, freq="M")

    periods = pd.period_range(end=final_month, periods=months, freq="M")
    plan_ids = [f"DENTAL-{index:03d}" for index in range(1, 7)]
    rows: list[dict[str, object]] = []

    for plan_index, plan_id in enumerate(plan_ids, start=1):
        for month in periods:
            rng = np.random.default_rng(_stable_seed(seed, plan_id, month))
            month_number = month.month
            month_index = (month.year - 2016) * 12 + month_number
            seasonal = 1.0 + 0.10 * np.sin(2 * np.pi * month_number / 12)
            trend = 1.0 + 0.0015 * max(month_index, 0)
            base_frequency = 0.075 + plan_index * 0.009
            expected_frequency = base_frequency * seasonal * trend

            member_months = int(rng.integers(2500, 12000))
            claim_count = int(rng.poisson(member_months * expected_frequency))
            average_claim_amount = (
                105.0 + 12.0 * plan_index + 0.12 * max(month_index, 0)
            ) * (1.0 + 0.06 * np.cos(2 * np.pi * month_number / 12))
            total_claim_amount = (
                float(
                    rng.gamma(
                        shape=claim_count,
                        scale=average_claim_amount,
                    )
                )
                if claim_count
                else 0.0
            )

            rows.append(
                {
                    "plan_id": plan_id,
                    "event_month": month.to_timestamp().date(),
                    "member_months": member_months,
                    "claim_count": claim_count,
                    "total_claim_amount": total_claim_amount,
                    "dataset_version": str(month),
                }
            )

    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def prepare_monthly_targets(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Validate monthly aggregates and calculate frequency and cost PMPM."""
    missing_columns = set(RAW_COLUMNS) - set(raw_data.columns)
    if missing_columns:
        raise ValueError(f"Missing source columns: {sorted(missing_columns)}")
    if raw_data.empty:
        raise ValueError("Source data cannot be empty")

    monthly = raw_data[RAW_COLUMNS].copy()
    monthly["event_month"] = pd.to_datetime(monthly["event_month"]).dt.to_period("M")
    numeric_columns = ["member_months", "claim_count", "total_claim_amount"]
    for column in numeric_columns:
        monthly[column] = pd.to_numeric(monthly[column], errors="raise")

    if monthly[numeric_columns].isna().any().any():
        raise ValueError("Source data contains null exposure or claim values")
    if (monthly["member_months"] <= 0).any():
        raise ValueError("member_months must be greater than zero")
    if (monthly[["claim_count", "total_claim_amount"]] < 0).any().any():
        raise ValueError("Claim counts and amounts cannot be negative")
    if monthly.duplicated(["plan_id", "event_month"]).any():
        raise ValueError("Source data has duplicate plan_id/event_month rows")

    monthly["claims_frequency_pmpm"] = (
        monthly["claim_count"] / monthly["member_months"]
    )
    monthly["claims_cost_pmpm"] = (
        monthly["total_claim_amount"] / monthly["member_months"]
    )
    return monthly.sort_values(["plan_id", "event_month"]).reset_index(drop=True)


def _features_for_origin(
    group: pd.DataFrame,
    plan_index: int,
    origin_index: int,
    forecast_horizon: int,
) -> dict[str, float | int]:
    target_month = group.iloc[origin_index]["event_month"] + forecast_horizon
    month_number = target_month.month
    features: dict[str, float | int] = {
        "plan_index": plan_index,
        "forecast_horizon": forecast_horizon,
        "target_month_sin": float(np.sin(2 * np.pi * month_number / 12)),
        "target_month_cos": float(np.cos(2 * np.pi * month_number / 12)),
    }

    for prefix, target_column in (
        ("frequency", "claims_frequency_pmpm"),
        ("cost_pmpm", "claims_cost_pmpm"),
    ):
        for lag in (1, 2, 3, 12):
            features[f"{prefix}_lag_{lag}"] = float(
                group.iloc[origin_index - lag + 1][target_column]
            )
    return features


def build_supervised_frame(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Create direct 1-3 month supervised forecast examples per plan."""
    monthly = prepare_monthly_targets(raw_data)
    rows: list[dict[str, object]] = []
    plan_ids = sorted(monthly["plan_id"].unique())

    for plan_index, plan_id in enumerate(plan_ids):
        group = (
            monthly[monthly["plan_id"] == plan_id]
            .sort_values("event_month")
            .reset_index(drop=True)
        )
        if _has_monthly_gaps(group["event_month"]):
            raise ValueError(f"Monthly history has gaps for plan {plan_id}")

        for origin_index in range(11, len(group) - 1):
            for horizon in (1, 2, 3):
                target_index = origin_index + horizon
                if target_index >= len(group):
                    continue
                row: dict[str, object] = _features_for_origin(
                    group, plan_index, origin_index, horizon
                )
                row["plan_id"] = plan_id
                row["origin_month"] = group.iloc[origin_index]["event_month"]
                row["target_month"] = group.iloc[target_index]["event_month"]
                for target_column in TARGET_COLUMNS:
                    row[target_column] = float(
                        group.iloc[target_index][target_column]
                    )
                rows.append(row)

    if not rows:
        raise ValueError("Not enough monthly history to build forecast examples")
    return pd.DataFrame(rows)


def chronological_split(
    supervised: pd.DataFrame,
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Period]:
    """Split forecast labels by time, keeping the latest periods for testing."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    months = sorted(supervised["target_month"].unique())
    split_index = int(len(months) * train_fraction)
    if split_index < 1 or split_index >= len(months):
        raise ValueError("Not enough target months for the requested split")

    test_start = months[split_index]
    train = supervised[supervised["target_month"] < test_start].copy()
    test = supervised[supervised["target_month"] >= test_start].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty train or test set")
    return train, test, test_start


def build_future_features(
    raw_data: pd.DataFrame,
    horizon_months: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the next N forecast rows per plan using latest observed history."""
    if horizon_months < 1:
        raise ValueError("horizon_months must be at least one")
    monthly = prepare_monthly_targets(raw_data)
    rows: list[dict[str, object]] = []
    plan_ids = sorted(monthly["plan_id"].unique())

    for plan_index, plan_id in enumerate(plan_ids):
        group = (
            monthly[monthly["plan_id"] == plan_id]
            .sort_values("event_month")
            .reset_index(drop=True)
        )
        if len(group) < 12:
            raise ValueError(f"At least 12 months are required for plan {plan_id}")
        if _has_monthly_gaps(group["event_month"]):
            raise ValueError(f"Monthly history has gaps for plan {plan_id}")

        origin_index = len(group) - 1
        origin_month = group.iloc[origin_index]["event_month"]
        for horizon in range(1, horizon_months + 1):
            row: dict[str, object] = _features_for_origin(
                group, plan_index, origin_index, horizon
            )
            row["plan_id"] = plan_id
            row["origin_month"] = origin_month
            row["forecast_month"] = origin_month + horizon
            rows.append(row)

    forecast_rows = pd.DataFrame(rows)
    return forecast_rows, forecast_rows[FEATURE_COLUMNS].copy()
