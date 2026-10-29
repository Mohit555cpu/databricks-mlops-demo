"""Utilities for the retail Databricks MLOps demo."""

def validate_predictions(predictions, min_rating=1.0, max_rating=5.0):
    """Validate model prediction values."""
    if not predictions:
        raise ValueError("Predictions cannot be empty")

    for value in predictions:
        if value is None:
            raise ValueError("Prediction cannot be null")
        if not min_rating <= value <= max_rating:
            raise ValueError("Prediction is outside the allowed range")

    return True
