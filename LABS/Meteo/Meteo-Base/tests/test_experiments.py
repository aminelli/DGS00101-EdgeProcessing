import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error

import meteo.experiments as experiments
from meteo.config import Settings
from meteo.experiments import (
    _add_recent_features,
    _split_frame,
    format_experiment_results,
    run_experiments,
)
from meteo.features import build_training_frame
from test_features import synthetic_hourly


def test_experiment_selection_and_saved_models_are_consistent(tmp_path):
    hourly = synthetic_hourly(150)
    settings = Settings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")
    report = run_experiments(hourly, settings)
    frame, columns = build_training_frame(hourly)
    train, validation, test = _split_frame(frame)
    assert train["date"].max() + pd.Timedelta(days=2) <= validation["date"].min()
    assert validation["date"].max() + pd.Timedelta(days=2) <= test["date"].min()

    artifact = joblib.load(settings.artifacts_dir / "experiments" / "experimental_models.joblib")
    features = _add_recent_features(test, columns)
    assert list(features.columns) == artifact["feature_columns"]
    for target, result in report["targets"].items():
        validation_scores = result["validation_candidates"]
        assert result["selected"] == min(
            validation_scores, key=lambda name: validation_scores[name]["mae"]
        )
        prediction = artifact["models"][target].predict(features)
        assert result["test_selected"]["mae"] == pytest.approx(
            mean_absolute_error(test[f"target_{target}"], prediction)
        )
        assert len(validation_scores[result["selected"]]["periods"]) == 3
        assert (settings.artifacts_dir / "experiments" / "plots" / f"{target}_test.png").is_file()
    rain = report["targets"]["precipitation_sum"]["rain_classification"]
    assert rain["available"]
    assert np.asarray(rain["confusion_matrix"]).sum() == len(test)
    assert "matrice test" in format_experiment_results(report)
    assert not settings.model_path.exists()


def test_experiment_rejects_missing_complete_day(tmp_path):
    hourly = synthetic_hourly(150)
    hourly = hourly.loc[hourly["timestamp"].dt.date != pd.Timestamp("2025-02-01").date()]
    with pytest.raises(ValueError, match="giornate complete mancanti"):
        run_experiments(hourly, Settings(artifacts_dir=tmp_path))


def test_model_selection_does_not_depend_on_test_targets(tmp_path, monkeypatch):
    hourly = synthetic_hourly(120)
    original_split = experiments._split_frame
    settings = Settings(artifacts_dir=tmp_path)
    original = run_experiments(hourly, settings)

    def split_with_modified_test(frame):
        train, validation, test = original_split(frame)
        test = test.copy()
        for column in ("target_temperature_2m_max", "target_precipitation_sum", "target_wind_speed_10m_max"):
            test[column] += 100
        return train, validation, test

    monkeypatch.setattr(experiments, "_split_frame", split_with_modified_test)
    changed = run_experiments(hourly, settings)
    for target in original["targets"]:
        assert changed["targets"][target]["selected"] == original["targets"][target]["selected"]
        for name, scores in original["targets"][target]["validation_candidates"].items():
            changed_scores = changed["targets"][target]["validation_candidates"][name]
            assert changed_scores["mae"] == pytest.approx(scores["mae"])
            assert changed_scores["rmse"] == pytest.approx(scores["rmse"])
        assert changed["targets"][target]["test_selected"]["mae"] != pytest.approx(
            original["targets"][target]["test_selected"]["mae"]
        )


def test_no_rain_in_training_reports_unavailable(tmp_path):
    hourly = synthetic_hourly(120)
    hourly["precipitation"] = 0.0
    report = run_experiments(hourly, Settings(artifacts_dir=tmp_path))
    rain = report["targets"]["precipitation_sum"]["rain_classification"]
    assert not rain["available"]
    assert "non contiene entrambe le classi" in rain["reason"]
