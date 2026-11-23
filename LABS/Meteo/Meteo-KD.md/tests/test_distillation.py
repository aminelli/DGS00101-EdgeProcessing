import numpy as np
import pandas as pd
import pytest
import json
from datetime import datetime, timezone

import meteo.national as national
from meteo.config import Settings
from meteo.data import HOURLY_VARIABLES
from meteo.distillation import forecast_student, train_distillation
from meteo.national import point_settings


def catalog_fixture():
    points = {
        "p001-1": {"id": "p001-1", "name": "Torino", "latitude": 45.0, "longitude": 7.7},
        "p002-1": {"id": "p002-1", "name": "Vercelli", "latitude": 45.3, "longitude": 8.4},
    }
    return {
        "points": points,
        "regions": [{"id": "01", "name": "Piemonte", "points": ["p001-1"]}],
        "provinces": [{"id": "001", "name": "Torino", "points": ["p001-1", "p002-1"]}],
    }


def save_hourly(settings, point, days=180):
    hours = pd.date_range("2024-01-01", periods=days * 24, freq="h", tz="UTC")
    values = np.arange(len(hours), dtype=float)
    frame = pd.DataFrame({
        "timestamp": hours, "location": point["name"],
        **{name: 10 + np.sin(values / (40 + i)) + i for i, name in enumerate(HOURLY_VARIABLES)},
    })
    local = point_settings(settings, point)
    local.processed_dir.mkdir(parents=True)
    (local.data_dir / "identity.json").write_text(json.dumps({
        "name": point["name"], "latitude": point["latitude"], "longitude": point["longitude"],
    }), encoding="utf-8")
    frame.to_csv(local.hourly_csv, index=False)
    return frame


def test_distilled_models_use_separate_temporal_test_and_province_points(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")
    catalog = catalog_fixture()
    for point in catalog["points"].values():
        save_hourly(settings, point)
    report = train_distillation(settings, catalog)
    assert set(report["territories"]) == {"region:01", "province:001"}
    result = report["territories"]["province:001"]
    assert result["test_samples"] > report["territories"]["region:01"]["test_samples"]
    assert set(result["targets"]["temperature_2m_max"]) == {
        "unit", "student", "supervised_student", "teacher", "persistence"
    }
    assert report["distillation_weight"] == 0.7
    assert result["student_size_bytes"] < report["teacher_size_bytes"]
    for point in catalog["points"]:
        forecast = forecast_student(settings, "province", "001", point)
        assert forecast["point"]["id"] == point
        assert forecast["forecast_date"] == "2024-06-30"
        assert forecast["student"]["precipitation_sum"] >= 0
    with pytest.raises(ValueError):
        forecast_student(settings, "region", "01", "p002-1")


def test_missing_day_is_rejected_before_training(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")
    catalog = catalog_fixture()
    for point in catalog["points"].values():
        frame = save_hourly(settings, point)
        local = point_settings(settings, point)
        frame = frame.drop(frame.index[24:48])
        frame.to_csv(local.hourly_csv, index=False)
    with pytest.raises(ValueError, match="non consecutive"):
        train_distillation(settings, catalog)


def test_missing_hourly_value_is_not_silently_aggregated(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")
    catalog = catalog_fixture()
    for point in catalog["points"].values():
        frame = save_hourly(settings, point)
        frame.loc[0, "precipitation"] = np.nan
        frame.to_csv(point_settings(settings, point).hourly_csv, index=False)
    with pytest.raises(ValueError, match="Valori orari"):
        train_distillation(settings, catalog)


@pytest.mark.parametrize("incomplete", [False, True])
def test_refresh_validates_new_daily_hours_before_replacing_csv(tmp_path, monkeypatch, incomplete):
    settings = Settings(data_dir=tmp_path / "data")
    point = catalog_fixture()["points"]["p001-1"]
    before = save_hourly(settings, point)

    class FrozenDatetime:
        @classmethod
        def now(cls, tz):
            return datetime(2024, 6, 30, tzinfo=timezone.utc)

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            hours = pd.date_range("2024-06-29", periods=23 if incomplete else 24, freq="h")
            return {"hourly": {
                "time": hours.strftime("%Y-%m-%dT%H:%M").tolist(),
                **{name: [1.0] * len(hours) for name in HOURLY_VARIABLES},
            }}

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, params, timeout):
            assert params["start_date"] == "2024-06-29"
            return Response()

    monkeypatch.setattr(national, "datetime", FrozenDatetime)
    monkeypatch.setattr(national.requests, "Session", Session)
    monkeypatch.setattr(national.time, "sleep", lambda seconds: None)
    if incomplete:
        with pytest.raises(RuntimeError, match="incompleta"):
            national.refresh_national_history(settings, {"points": {point["id"]: point}})
    else:
        national.refresh_national_history(settings, {"points": {point["id"]: point}})
    result = national.read_hourly_csv(point_settings(settings, point).hourly_csv)
    assert len(result) == len(before) + (0 if incomplete else 24)


def test_national_fetch_resumes_cached_chunks_with_fixed_dates_and_daily_budget(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "data")
    point = catalog_fixture()["points"]["p001-1"]
    catalog = {"points": {point["id"]: point}, "regions": [], "provinces": []}
    calls = []

    class FrozenDatetime(datetime):
        day = 30

        @classmethod
        def now(cls, tz):
            return cls(2024, 6, cls.day, tzinfo=timezone.utc)

    def fake_get(self, url, **kwargs):
        params = kwargs["params"]
        calls.append(params["start_date"])
        times = pd.date_range(params["start_date"], periods=24, freq="h")
        payload = {"latitude": point["latitude"], "longitude": point["longitude"], "hourly": {
            "time": times.strftime("%Y-%m-%dT%H:%M").tolist(),
            **{name: [1.0] * 24 for name in HOURLY_VARIABLES},
        }}
        response = national.requests.Response()
        response.status_code = 200
        response._content = json.dumps(payload).encode()
        return response

    def fake_fetch(local, session, end_date):
        assert end_date.isoformat() == "2024-06-29"
        local.raw_dir.mkdir(parents=True, exist_ok=True)
        for day in ("2024-06-28", "2024-06-29"):
            result = session.get("https://example.invalid", params={"start_date": day, "end_date": day})
            (local.raw_dir / f"archive_{day}_{day}.json").write_text(json.dumps(result.json()), encoding="utf-8")

    monkeypatch.setattr(national, "datetime", FrozenDatetime)
    monkeypatch.setattr(national.requests.Session, "get", fake_get)
    monkeypatch.setattr(national, "fetch_history", fake_fetch)
    monkeypatch.setattr(national.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="Budget sulle 24 ore"):
        national.fetch_national_history(settings, catalog, daily_budget=1)
    assert calls == ["2024-06-28"]
    FrozenDatetime.day = 1
    class NextMonthDatetime(FrozenDatetime):
        @classmethod
        def now(cls, tz):
            return cls(2024, 7, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(national, "datetime", NextMonthDatetime)
    national.fetch_national_history(settings, catalog, daily_budget=1)
    assert calls == ["2024-06-28", "2024-06-29"]
