"""Acquisizione Open-Meteo e controlli dei dati meteorologici."""

from __future__ import annotations

import calendar
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from meteo.config import Settings

LOGGER = logging.getLogger(__name__)
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
]
DAILY_TARGETS = [
    "temperature_2m_max",
    "precipitation_sum",
    "wind_speed_10m_max",
]


def _request_json(
    url: str, params: dict[str, Any], timeout: int
) -> dict[str, Any]:
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Richiesta Open-Meteo fallita: {exc}") from exc
    try:
        payload = response.json()
    except requests.JSONDecodeError as exc:
        raise RuntimeError("Open-Meteo ha restituito una risposta JSON non valida") from exc
    if not isinstance(payload, dict) or "error" in payload:
        reason = payload.get("reason", "risposta API non valida") if isinstance(payload, dict) else "risposta API non valida"
        raise RuntimeError(f"Errore Open-Meteo: {reason}")
    return payload


def fetch_history(settings: Settings, session: requests.Session | None = None) -> pd.DataFrame:
    """Scarica il periodo configurato, dieci anni per default, in blocchi da 90 giorni."""
    end_date = datetime.now(timezone.utc).date() - timedelta(days=1)
    target_year = end_date.year - settings.history_years
    start_date = end_date.replace(
        year=target_year,
        day=min(end_date.day, calendar.monthrange(target_year, end_date.month)[1]),
    ) + timedelta(days=1)
    cursor = start_date
    frames: list[pd.DataFrame] = []
    settings.raw_dir.mkdir(parents=True, exist_ok=True)

    while cursor <= end_date:
        chunk_end = min(cursor + timedelta(days=89), end_date)
        params = {
            "latitude": settings.latitude,
            "longitude": settings.longitude,
            "start_date": cursor.isoformat(),
            "end_date": chunk_end.isoformat(),
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": settings.timezone,
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "precipitation_unit": "mm",
        }
        if session is None:
            payload = _request_json(ARCHIVE_URL, params, settings.request_timeout_seconds)
        else:
            try:
                response = session.get(ARCHIVE_URL, params=params, timeout=settings.request_timeout_seconds)
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, requests.JSONDecodeError) as exc:
                raise RuntimeError(f"Download storico {cursor}–{chunk_end} fallito: {exc}") from exc
            if not isinstance(payload, dict) or "error" in payload:
                reason = payload.get("reason", "risposta API non valida") if isinstance(payload, dict) else "risposta API non valida"
                raise RuntimeError(f"Errore Open-Meteo per {cursor}–{chunk_end}: {reason}")

        raw_path = settings.raw_dir / f"archive_{cursor.isoformat()}_{chunk_end.isoformat()}.json"
        raw_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        hourly = payload.get("hourly")
        if not isinstance(hourly, dict) or "time" not in hourly:
            raise RuntimeError(f"Risposta storica incompleta per {cursor}–{chunk_end}: manca hourly.time")
        time_values = hourly["time"]
        expected_hours = (chunk_end - cursor).days * 24 + 24
        if len(time_values) != expected_hours:
            raise RuntimeError(
                f"Blocco {cursor}–{chunk_end} incompleto: "
                f"attese {expected_hours} ore, ricevute {len(time_values)}"
            )
        expected_variables = set(HOURLY_VARIABLES)
        missing = expected_variables - set(hourly)
        if missing:
            raise RuntimeError(f"Risposta storica incompleta: mancano {sorted(missing)}")
        if any(len(hourly[name]) != len(time_values) for name in HOURLY_VARIABLES):
            raise RuntimeError(f"Colonne con lunghezze diverse nel blocco {cursor}–{chunk_end}")

        frame = pd.DataFrame({name: hourly[name] for name in HOURLY_VARIABLES})
        frame.insert(0, "timestamp", pd.to_datetime(time_values, utc=True, errors="coerce"))
        frame["location"] = settings.location_name
        if frame["timestamp"].isna().any():
            raise RuntimeError(f"Timestamp non validi nel blocco {cursor}–{chunk_end}")
        frames.append(frame)
        LOGGER.info("Scaricati dati orari dal %s al %s (%d righe)", cursor, chunk_end, len(frame))
        cursor = chunk_end + timedelta(days=1)

    result = pd.concat(frames, ignore_index=True)
    result = validate_hourly_data(result)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(settings.hourly_csv, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    return result


def validate_hourly_data(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "location", *HOURLY_VARIABLES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Colonne obbligatorie mancanti: {sorted(missing)}")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(
        result["timestamp"], format="mixed", utc=True, errors="coerce"
    )
    if result["timestamp"].isna().any():
        raise ValueError("Il dataset contiene timestamp non validi")
    if result["location"].isna().any():
        raise ValueError("Il dataset contiene località mancanti")
    duplicate_count = int(result["timestamp"].duplicated().sum())
    if duplicate_count:
        LOGGER.warning("Rimosse %d righe con timestamp duplicato", duplicate_count)
    result = result.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if result.empty:
        raise ValueError("Il dataset non contiene osservazioni")
    for column in HOURLY_VARIABLES:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    missing = result[HOURLY_VARIABLES].isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        LOGGER.warning("Valori orari mancanti per colonna: %s", missing.to_dict())
    return result.reset_index(drop=True)


def fetch_nwp_forecast(settings: Settings) -> pd.DataFrame:
    """Ottiene la previsione giornaliera NWP operativa per la località configurata."""
    payload = _request_json(
        FORECAST_URL,
        {
            "latitude": settings.latitude,
            "longitude": settings.longitude,
            "daily": ",".join(DAILY_TARGETS),
            "timezone": settings.timezone,
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "precipitation_unit": "mm",
            "forecast_days": 7,
        },
        settings.request_timeout_seconds,
    )
    daily = payload.get("daily")
    if not isinstance(daily, dict) or "time" not in daily:
        raise RuntimeError("La risposta NWP non contiene la sezione daily")
    missing = set(DAILY_TARGETS) - set(daily)
    if missing:
        raise RuntimeError(f"Risposta NWP incompleta: mancano {sorted(missing)}")
    if any(len(daily[name]) != len(daily["time"]) for name in DAILY_TARGETS):
        raise RuntimeError("La risposta NWP contiene serie di lunghezze diverse")
    result = pd.DataFrame({name: daily[name] for name in DAILY_TARGETS})
    result.insert(0, "date", pd.to_datetime(daily["time"], errors="coerce"))
    if result["date"].isna().any():
        raise RuntimeError("La risposta NWP contiene date non valide")
    if result[DAILY_TARGETS].isna().any().any():
        raise RuntimeError("La risposta NWP contiene valori giornalieri mancanti")
    result["source"] = "Open-Meteo forecast (NWP)"
    return result


def read_hourly_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    return validate_hourly_data(frame)
