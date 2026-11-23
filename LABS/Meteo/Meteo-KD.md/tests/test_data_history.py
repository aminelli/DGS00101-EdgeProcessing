from datetime import datetime, timezone

import pandas as pd

import meteo.data as data
from meteo.config import Settings, load_settings


class FrozenDatetime:
    @classmethod
    def now(cls, tz):
        return datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc)


class FakeResponse:
    def __init__(self, params):
        self.params = params

    def raise_for_status(self):
        pass

    def json(self):
        times = pd.date_range(
            self.params["start_date"],
            pd.Timestamp(self.params["end_date"]) + pd.Timedelta(hours=23),
            freq="h",
        ).strftime("%Y-%m-%dT%H:%M").tolist()
        return {"hourly": {"time": times, **{name: [1.0] * len(times) for name in data.HOURLY_VARIABLES}}}


class FakeSession:
    def __init__(self):
        self.requests = []

    def get(self, url, params, timeout):
        self.requests.append(params.copy())
        return FakeResponse(params)


def test_default_history_downloads_ten_complete_calendar_years(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "datetime", FrozenDatetime)
    session = FakeSession()
    settings = Settings(data_dir=tmp_path)
    result = data.fetch_history(settings, session=session)

    assert settings.history_years == 10
    assert load_settings().history_years == 10
    assert session.requests[0]["start_date"] == "2016-11-23"
    assert session.requests[-1]["end_date"] == "2026-11-22"
    assert len(result) == 87648
    assert result["timestamp"].is_unique
    assert result["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all()
    assert settings.hourly_csv.is_file()


def test_history_years_remains_configurable(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "datetime", FrozenDatetime)
    session = FakeSession()
    result = data.fetch_history(Settings(data_dir=tmp_path, history_years=1), session=session)
    assert session.requests[0]["start_date"] == "2025-11-23"
    assert session.requests[-1]["end_date"] == "2026-11-22"
    assert len(result) == 365 * 24
