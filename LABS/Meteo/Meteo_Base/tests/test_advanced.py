import numpy as np
import pandas as pd
import pytest

import meteo.nwp_history as nwp_history
from meteo.advanced import _build_frame, format_advanced_results, run_advanced_experiment
from meteo.config import Settings
from meteo.nwp_history import NWP_MODEL, PREVIOUS_RUNS_URL, read_nwp_history
from test_features import synthetic_hourly


def _nwp_frame(hourly, settings):
    dates = pd.to_datetime(hourly["timestamp"], utc=True).dt.floor("D").drop_duplicates()
    dates = dates.reset_index(drop=True)
    return pd.DataFrame({
        "target_date": dates,
        "latest_nominal_forecast_time": dates - pd.Timedelta(hours=49),
        "nwp_temperature_2m_max": 15.0,
        "nwp_wind_speed_10m_max": 10.0,
        "latitude": settings.latitude,
        "longitude": settings.longitude,
        "model": NWP_MODEL,
        "lead_hours": 72,
        "source": PREVIOUS_RUNS_URL,
    })


def _write_nwp(settings, frame):
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(settings.processed_dir / "nwp_previous_runs_daily.csv", index=False)


def test_nwp_download_aggregates_complete_days_and_drops_incomplete(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path)

    def fake_request(url, params, timeout):
        assert params["models"] == NWP_MODEL
        assert "temperature_2m_previous_day3" in params["hourly"]
        times = pd.date_range(params["start_date"], periods=48, freq="h")
        temp = [float(index) for index in range(48)]
        temp[30] = None
        return {"hourly": {
            "time": times.strftime("%Y-%m-%dT%H:%M").tolist(),
            "temperature_2m_previous_day3": temp,
            "wind_speed_10m_previous_day3": [2.0] * 48,
        }}

    monkeypatch.setattr(nwp_history, "_request_json", fake_request)
    result = nwp_history.fetch_nwp_history(
        settings, pd.Timestamp("2025-01-01").date(), pd.Timestamp("2025-01-02").date()
    )
    assert len(result) == 1
    assert result.iloc[0]["nwp_temperature_2m_max"] == 23
    assert len(read_nwp_history(settings)) == 1


def test_nwp_rejects_bad_provenance_and_future_nominal_time(tmp_path):
    settings = Settings(data_dir=tmp_path)
    frame = _nwp_frame(synthetic_hourly(2), settings)
    frame.loc[0, "latest_nominal_forecast_time"] = frame.loc[0, "target_date"]
    _write_nwp(settings, frame)
    with pytest.raises(ValueError, match="Anticipo nominale"):
        read_nwp_history(settings)
    frame = _nwp_frame(synthetic_hourly(2), settings)
    frame.loc[0, "latitude"] = 0.0
    _write_nwp(settings, frame)
    with pytest.raises(ValueError, match="coordinate"):
        read_nwp_history(settings)


def test_advanced_features_use_only_hours_up_to_feature_day(tmp_path):
    hourly = synthetic_hourly(30)
    settings = Settings(data_dir=tmp_path)
    nwp = _nwp_frame(hourly, settings)
    frame, columns, coverage = _build_frame(hourly, nwp)
    day = frame.iloc[0]["date"]
    changed = hourly.copy()
    changed.loc[changed["timestamp"] > day + pd.Timedelta(hours=23), "temperature_2m"] += 50
    changed_frame, _, _ = _build_frame(changed, nwp)
    before = frame.loc[frame["date"] == day, columns].to_numpy(dtype=float)
    after = changed_frame.loc[changed_frame["date"] == day, columns].to_numpy(dtype=float)
    np.testing.assert_allclose(before, after)
    assert coverage["nwp_aligned_examples"] == len(frame)


def test_advanced_experiment_rolling_selection_and_report(tmp_path):
    hourly = synthetic_hourly(240)
    settings = Settings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")
    _write_nwp(settings, _nwp_frame(hourly, settings))
    report = run_advanced_experiment(hourly, settings)
    assert report["split"]["test_samples"] > 0
    assert pd.Timestamp(report["split"]["pretest_last_target_date"]) <= pd.Timestamp(
        report["split"]["test_first_feature_date"]
    )
    for fold in report["split"]["folds"]:
        assert pd.Timestamp(fold["train_last_target_date"]) <= pd.Timestamp(fold["validation_first_feature_date"])
    for target, result in report["targets"].items():
        candidates = result["rolling_candidates"]
        assert "hist_gradient_boosting" in candidates
        assert result["selected"] == min(candidates, key=lambda name: candidates[name]["mae"])
        assert len(candidates[result["selected"]]["folds"]) == 3
        assert sum(group["samples"] for group in result["test_by_season"].values()) == report["split"]["test_samples"]
        assert result["test_nwp"]["mae"] >= 0
        assert (settings.artifacts_dir / "advanced_experiments" / "plots" / f"{target}_test.png").is_file()
    assert sum(
        item["samples"]
        for item in report["targets"]["wind_speed_10m_max"]["test_by_observed_wind_intensity"].values()
    ) == report["split"]["test_samples"]
    assert "MAE NWP" in format_advanced_results(report)
    assert not settings.model_path.exists()
