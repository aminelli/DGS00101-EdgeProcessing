"""Acquisizione delle previsioni NWP archiviate a anticipo fisso."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from meteo.config import Settings
from meteo.data import _request_json

LOGGER = logging.getLogger(__name__)
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
NWP_MODEL = "gfs_seamless"
NWP_LEAD_HOURS = 72
NWP_VARIABLES = {
    "temperature_2m_previous_day3": "nwp_temperature_2m_max",
    "wind_speed_10m_previous_day3": "nwp_wind_speed_10m_max",
}


def fetch_nwp_history(
    settings: Settings, start: date, end: date,
) -> pd.DataFrame:
    """Scarica previsioni orarie GFS, le aggrega e conserva la provenienza."""
    if start > end:
        raise ValueError("La data iniziale NWP deve precedere la data finale")
    if settings.timezone != "UTC":
        raise ValueError("Le previsioni NWP storiche richiedono timezone UTC")
    cursor = start
    blocks: list[pd.DataFrame] = []
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=89), end)
        params = {
            "latitude": settings.latitude,
            "longitude": settings.longitude,
            "start_date": cursor.isoformat(),
            "end_date": chunk_end.isoformat(),
            "hourly": ",".join(NWP_VARIABLES),
            "models": NWP_MODEL,
            "timezone": "UTC",
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
        }
        payload = _request_json(PREVIOUS_RUNS_URL, params, settings.request_timeout_seconds)
        hourly = payload.get("hourly")
        if not isinstance(hourly, dict) or not {"time", *NWP_VARIABLES}.issubset(hourly):
            raise RuntimeError(f"Risposta NWP incompleta per {cursor} - {chunk_end}")
        expected = pd.date_range(
            pd.Timestamp(cursor, tz="UTC"),
            pd.Timestamp(chunk_end, tz="UTC") + pd.Timedelta(hours=23),
            freq="h",
        )
        timestamps = pd.to_datetime(hourly["time"], utc=True, errors="coerce")
        if len(timestamps) != len(expected) or not timestamps.equals(expected):
            raise RuntimeError(f"Timestamp NWP mancanti, duplicati o fuori ordine per {cursor} - {chunk_end}")
        columns: dict[str, object] = {"timestamp": timestamps}
        for source in NWP_VARIABLES:
            if len(hourly[source]) != len(expected):
                raise RuntimeError(f"Serie NWP {source} incompleta per {cursor} - {chunk_end}")
            columns[source] = pd.to_numeric(pd.Series(hourly[source]), errors="coerce").to_numpy()
        blocks.append(pd.DataFrame(columns))
        LOGGER.info("Scaricata previsione NWP dal %s al %s", cursor, chunk_end)
        cursor = chunk_end + timedelta(days=1)

    hours = pd.concat(blocks, ignore_index=True)
    hours["target_date"] = hours["timestamp"].dt.floor("D")
    for column in NWP_VARIABLES:
        hours[column] = hours[column].replace([np.inf, -np.inf], np.nan)
    grouped = hours.groupby("target_date", sort=True)
    complete = grouped[list(NWP_VARIABLES)].count().eq(24).all(axis=1)
    daily = grouped[list(NWP_VARIABLES)].max().rename(columns=NWP_VARIABLES)
    missing_days = int((~complete).sum())
    if missing_days:
        LOGGER.warning("Escluse %d giornate NWP con previsioni orarie incomplete", missing_days)
    daily = daily.loc[complete].reset_index()
    if daily.empty:
        raise RuntimeError("Nessuna giornata NWP completa disponibile nel periodo richiesto")
    daily["latest_nominal_forecast_time"] = (
        daily["target_date"] + pd.Timedelta(hours=23 - NWP_LEAD_HOURS)
    )
    daily["latitude"] = settings.latitude
    daily["longitude"] = settings.longitude
    daily["model"] = NWP_MODEL
    daily["lead_hours"] = NWP_LEAD_HOURS
    daily["source"] = PREVIOUS_RUNS_URL
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    path = settings.processed_dir / "nwp_previous_runs_daily.csv"
    daily.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    LOGGER.info("Salvate %d giornate NWP in %s", len(daily), path)
    return daily


def read_nwp_history(settings: Settings) -> pd.DataFrame:
    path = settings.processed_dir / "nwp_previous_runs_daily.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Archivio NWP non trovato: {path}. Eseguire prima fetch-nwp-history."
        )
    frame = pd.read_csv(path)
    required = {
        "target_date", "latest_nominal_forecast_time", "latitude", "longitude",
        "model", "lead_hours", "source", *NWP_VARIABLES.values(),
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Colonne NWP mancanti: {sorted(missing)}")
    for column in ("target_date", "latest_nominal_forecast_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if frame.empty or frame[["target_date", "latest_nominal_forecast_time"]].isna().any().any():
        raise ValueError("Date NWP mancanti o non valide")
    if frame["target_date"].duplicated().any():
        raise ValueError("Date target NWP duplicate")
    if not frame["target_date"].eq(frame["target_date"].dt.floor("D")).all():
        raise ValueError("Le date target NWP devono iniziare a mezzanotte UTC")
    expected_time = frame["target_date"] + pd.Timedelta(hours=23 - NWP_LEAD_HOURS)
    if not frame["latest_nominal_forecast_time"].eq(expected_time).all():
        raise ValueError("Anticipo nominale NWP incoerente con la data target")
    if (
        not frame["model"].eq(NWP_MODEL).all()
        or not frame["lead_hours"].eq(NWP_LEAD_HOURS).all()
        or not frame["source"].eq(PREVIOUS_RUNS_URL).all()
        or not np.isclose(frame["latitude"], settings.latitude, atol=1e-5, rtol=0).all()
        or not np.isclose(frame["longitude"], settings.longitude, atol=1e-5, rtol=0).all()
    ):
        raise ValueError("Provenienza, coordinate o anticipo dell'archivio NWP non compatibili")
    for column in NWP_VARIABLES.values():
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[list(NWP_VARIABLES.values())].to_numpy(dtype=float)).all():
        raise ValueError("L'archivio NWP contiene valori mancanti o non finiti")
    return frame.sort_values("target_date").reset_index(drop=True)


def default_nwp_end_date() -> date:
    return datetime.now(timezone.utc).date() - timedelta(days=1)
