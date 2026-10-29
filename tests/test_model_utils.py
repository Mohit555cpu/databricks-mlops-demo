"""Tests for src.model_utils."""


import pytest
from src.retail.model_utils import validate_predictions


def test_valid_predictions():
    assert validate_predictions([1.2, 3.0, 4.8]) is True


def test_empty_predictions_fail():
    with pytest.raises(ValueError):
        validate_predictions([])


def test_null_prediction_fails():
    with pytest.raises(ValueError):
        validate_predictions([3.0, None])


def test_out_of_range_prediction_fails():
    with pytest.raises(ValueError):
        validate_predictions([6.0])
