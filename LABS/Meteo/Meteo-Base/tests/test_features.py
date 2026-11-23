from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from meteo.config import Settings
from meteo.data import HOURLY_VARIABLES, validate_hourly_data
from meteo.features import aggregate_daily, build_training_frame
from meteo.training import format_test_results, train_models


def synthetic_hourly(days: int = 20) -> pd.DataFrame:
    rows = []
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for hour in range(days * 24):
        timestamp = start + timedelta(hours=hour)
        daily_phase = timestamp.hour / 24 * 2 * np.pi
        values = {
            "temperature_2m": 10 + timestamp.day * 0.1 + np.sin(daily_phase) * 4,
            "relative_humidity_2m": 70 + np.cos(daily_phase) * 10,
            "dew_point_2m": 5 + np.sin(daily_phase) * 2,
            "precipitation": 0.1 if timestamp.day % 5 == 0 and timestamp.hour == 4 else 0,
            "pressure_msl": 1013 + np.cos(timestamp.day / 4),
            "cloud_cover": 40 + np.sin(timestamp.day / 3) * 20,
            "wind_speed_10m": 8 + np.cos(daily_phase) * 2,
            "wind_direction_10m": 180,
            "shortwave_radiation": max(0, np.sin(daily_phase) * 300),
        }
        rows.append({"timestamp": timestamp, "location": "Roma", **values})
    return pd.DataFrame(rows, columns=["timestamp", "location", *HOURLY_VARIABLES])


def test_aggregation_discards_incomplete_days():
    hourly = synthetic_hourly(3).iloc[1:].copy()
    daily = aggregate_daily(hourly)
    assert len(daily) == 2
    assert daily["date"].is_monotonic_increasing


def test_training_frame_targets_next_day_and_features_are_current_or_past():
    hourly = synthetic_hourly(20)
    frame, feature_columns = build_training_frame(hourly)
    assert "temperature_2m_max" in feature_columns
    first = frame.iloc[0]
    first_date = pd.Timestamp(first["date"])
    expected_forecast_max = aggregate_daily(hourly).set_index("date").loc[
        first_date + pd.Timedelta(days=2), "temperature_2m_max"
    ]
    assert first["target_temperature_2m_max"] == expected_forecast_max
    assert not any(name.startswith("target_") for name in feature_columns)


def test_validation_sorts_deduplicates_and_rejects_bad_timestamp():
    hourly = synthetic_hourly(2)
    frame = pd.concat([hourly.iloc[::-1], hourly.iloc[[0]]], ignore_index=True)
    checked = validate_hourly_data(frame)
    assert checked["timestamp"].is_monotonic_increasing
    assert checked["timestamp"].is_unique
    frame["timestamp"] = frame["timestamp"].astype(str)
    frame.loc[0, "timestamp"] = "not-a-date"
    with pytest.raises(ValueError, match="timestamp"):
        validate_hourly_data(frame)


def test_training_uses_chronological_splits_and_persists_metrics(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
    )
    metrics = train_models(synthetic_hourly(150), settings)
    assert metrics["sample_counts"]["train"] > metrics["sample_counts"]["validation"]
    assert set(metrics["targets"]) == {
        "temperature_2m_max",
        "precipitation_sum",
        "wind_speed_10m_max",
    }
    assert "validation" in metrics["targets"]["temperature_2m_max"]
    assert settings.model_path.is_file()
    assert settings.metrics_path.is_file()
    for target in metrics["targets"]:
        plot = settings.artifacts_dir / "plots" / f"{target}_test.png"
        assert plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    table = format_test_results(metrics)
    assert "MAE" in table and "RMSE" in table and "R2" in table
    assert "Matrice di confusione non applicabile" in table
    for target, result in metrics["targets"].items():
        assert target in table
        assert f"{result['machine_learning']['mae']:.3f}" in table
        assert f"{result['persistence_baseline']['rmse']:.3f}" in table
