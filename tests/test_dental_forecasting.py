import pandas as pd
import pytest

from src.dental.dental_forecasting_utils import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    build_future_features,
    build_supervised_frame,
    chronological_split,
    filter_snapshot_window,
    generate_synthetic_claims,
    prepare_monthly_targets,
    resolve_cutoff_month,
    snapshot_id_for_cutoff,
)


def test_synthetic_data_is_reproducible_and_has_required_monthly_rows():
    first = generate_synthetic_claims(months=36, end_month="2026-09")
    second = generate_synthetic_claims(months=36, end_month="2026-09")

    pd.testing.assert_frame_equal(first, second)
    assert first["plan_id"].nunique() == 6
    assert first.groupby("plan_id")["event_month"].nunique().eq(36).all()


def test_monthly_cutoff_uses_last_complete_month_in_ist():
    cutoff = resolve_cutoff_month(
        reference_timestamp="2026-11-01T00:00:00+00:00"
    )

    assert str(cutoff) == "2026-10"
    assert snapshot_id_for_cutoff(cutoff) == "dental-claims-2026-10"


def test_snapshot_window_keeps_exact_120_months_per_plan():
    raw = generate_synthetic_claims(months=120, end_month="2026-09")

    snapshot = filter_snapshot_window(raw, cutoff_month="2026-09")

    assert snapshot.groupby("plan_id")["event_month"].nunique().eq(120).all()
    assert pd.to_datetime(snapshot["event_month"]).max().strftime("%Y-%m") == "2026-09"
    assert pd.to_datetime(snapshot["event_month"]).min().strftime("%Y-%m") == "2016-10"


def test_snapshot_window_rejects_missing_months():
    raw = generate_synthetic_claims(months=36, end_month="2026-09")
    missing_month = raw[
        ~(
            (pd.to_datetime(raw["event_month"]).dt.to_period("M") == "2025-01")
            & (raw["plan_id"] == "DENTAL-001")
        )
    ]

    with pytest.raises(ValueError, match="every month"):
        filter_snapshot_window(
            missing_month,
            cutoff_month="2026-09",
            lookback_months=36,
        )


def test_monthly_targets_calculate_frequency_and_cost_per_member_month():
    raw = generate_synthetic_claims(months=36, end_month="2026-09")
    monthly = prepare_monthly_targets(raw)

    expected_frequency = raw["claim_count"] / raw["member_months"]
    expected_cost = raw["total_claim_amount"] / raw["member_months"]
    assert monthly["claims_frequency_pmpm"].tolist() == pytest.approx(
        expected_frequency.tolist()
    )
    assert monthly["claims_cost_pmpm"].tolist() == pytest.approx(
        expected_cost.tolist()
    )


def test_supervised_rows_have_three_horizons_and_both_targets():
    raw = generate_synthetic_claims(months=36, end_month="2026-09")
    supervised = build_supervised_frame(raw)

    assert set(supervised["forecast_horizon"].unique()) == {1, 2, 3}
    assert "plan_id" in supervised.columns
    assert set(TARGET_COLUMNS).issubset(supervised.columns)
    assert set(FEATURE_COLUMNS).issubset(supervised.columns)


def test_chronological_split_places_all_test_months_after_training():
    raw = generate_synthetic_claims(months=48, end_month="2026-09")
    supervised = build_supervised_frame(raw)

    train, test, test_start = chronological_split(supervised, 0.8)

    assert train["target_month"].max() < test_start
    assert test["target_month"].min() == test_start
    assert len(test["target_month"].unique()) > 0


def test_future_features_cover_three_months_for_each_plan():
    raw = generate_synthetic_claims(months=36, end_month="2026-09")

    forecast_rows, features = build_future_features(raw, horizon_months=3)

    assert len(forecast_rows) == 6 * 3
    assert list(features.columns) == FEATURE_COLUMNS
    horizons_per_plan = forecast_rows.groupby("plan_id")["forecast_horizon"].apply(
        lambda values: sorted(values.tolist())
    )
    assert horizons_per_plan.map(lambda values: values == [1, 2, 3]).all()


def test_duplicate_plan_months_are_rejected():
    raw = generate_synthetic_claims(months=36, end_month="2026-09")
    duplicated = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        prepare_monthly_targets(duplicated)
