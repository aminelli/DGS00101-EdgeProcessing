import json

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

import meteo.web as web
from meteo.config import Settings
from meteo.data import HOURLY_VARIABLES


@pytest.fixture
def settings(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")
    monkeypatch.setattr(web, "SETTINGS", settings)
    return settings


def hourly_data():
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=72, freq="h", tz="UTC"),
        "location": "Roma",
        **{name: [1.0] * 72 for name in HOURLY_VARIABLES},
    })


def test_history_selects_calendar_days_and_aggregates(settings, monkeypatch):
    monkeypatch.setattr(web, "read_hourly_csv", lambda path: hourly_data())
    result = web.history(days=2)
    assert [row["date"] for row in result["records"]] == ["2025-01-02", "2025-01-03"]
    assert result["records"][0]["precipitation_sum"] == 24
    assert result["timezone"] == "UTC"


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_history_rejects_nonfinite_daily_values(settings, monkeypatch, value):
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01"], utc=True),
        **{target: [value] for target in web.DAILY_TARGETS},
    })
    monkeypatch.setattr(web, "read_hourly_csv", lambda path: hourly_data())
    monkeypatch.setattr(web, "aggregate_daily", lambda hourly: frame)
    with pytest.raises(HTTPException) as exc:
        web.history(days=1)
    assert exc.value.status_code == 422


def test_history_missing_file_is_reported(settings):
    with pytest.raises(HTTPException) as exc:
        web.history(days=90)
    assert exc.value.status_code == 503


def test_evaluation_reports_and_plot_allowlist(settings):
    settings.artifacts_dir.mkdir()
    report = {"targets": {"temperature_2m_max": {"unit": "°C"}}}
    (settings.artifacts_dir / "metrics.json").write_text(json.dumps(report), encoding="utf-8")
    assert web.evaluation("operativo") == report
    plot_dir = settings.artifacts_dir / "plots"
    plot_dir.mkdir()
    plot = plot_dir / "temperature_2m_max_test.png"
    plot.write_bytes(b"png")
    assert web.evaluation_plot("operativo", "temperature_2m_max").path == plot
    for report_name, target in [("invalid", "temperature_2m_max"), ("operativo", "../metrics")]:
        with pytest.raises(HTTPException) as exc:
            web.evaluation_plot(report_name, target)
        assert exc.value.status_code == 404
    with pytest.raises(HTTPException) as exc:
        web.evaluation("sperimentale")
    assert exc.value.status_code == 404


@pytest.mark.parametrize("contents", ["{", "[]", '{"targets": []}'])
def test_invalid_evaluation_is_reported(settings, contents):
    settings.artifacts_dir.mkdir()
    (settings.artifacts_dir / "metrics.json").write_text(contents, encoding="utf-8")
    with pytest.raises(HTTPException) as exc:
        web.evaluation("operativo")
    assert exc.value.status_code == 500
