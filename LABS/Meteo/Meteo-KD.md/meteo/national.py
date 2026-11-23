"""Catalogo territoriale e acquisizione separata dal modello di Roma."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
import os
import re
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from meteo.config import Settings
from meteo.data import ARCHIVE_URL, HOURLY_VARIABLES, fetch_history, read_hourly_csv, validate_hourly_data

LOGGER = logging.getLogger(__name__)
PROVINCES_URL = "https://raw.githubusercontent.com/guglielmo/geojson-italy/2026.1/geojson/limits_IT_provinces.geojson"
DPC_URL = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province-20250108.csv"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

SPECIAL_CAPITALS = {
    "040": ["Forlì", "Cesena"],
    "041": ["Pesaro", "Urbino"],
    "045": ["Massa"],
    "103": ["Verbania"],
    "108": ["Monza"],
    "110": ["Barletta", "Andria", "Trani"],
    "112": ["Sassari"],
    "113": ["Olbia", "Tempio Pausania"],
    "114": ["Nuoro"],
    "115": ["Oristano"],
    "116": ["Lanusei", "Tortolì"],
    "117": ["Sanluri", "Villacidro"],
    "118": ["Cagliari"],
    "119": ["Carbonia", "Iglesias"],
}

REGION_CAPITAL_PROVINCE = {
    "01": "001", "02": "007", "03": "015", "04": "022", "05": "027",
    "06": "032", "07": "010", "08": "037", "09": "048", "10": "054",
    "11": "042", "12": "058", "13": "066", "14": "070", "15": "063",
    "16": "072", "17": "076", "18": "079", "19": "082", "20": "118",
}


class NationalBudgetExceeded(RuntimeError):
    """Budget locale raggiunto; l'acquisizione si può riprendere."""


def _get(url: str, *, params: dict | None = None, timeout: int = 60) -> requests.Response:
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Acquisizione da {url} fallita: {exc}") from exc
    return response


def national_dir(settings: Settings) -> Path:
    return settings.data_dir / "national"


def active_distillation_dir(settings: Settings) -> Path:
    root = settings.artifacts_dir / "distillation"
    pointer = root / "current"
    if not pointer.is_file():
        return root
    run_id = pointer.read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("Identificativo della versione dei modelli nazionali non valido")
    return root / "runs" / run_id


def load_catalog(settings: Settings, path: Path | None = None) -> dict:
    path = national_dir(settings) / "catalog.json" if path is None else path
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError("Catalogo nazionale mancante: eseguire national-catalog.") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Catalogo nazionale non leggibile: {path}") from exc
    if not isinstance(result, dict) or not all(key in result for key in ("points", "regions", "provinces")):
        raise ValueError("Formato del catalogo nazionale non valido")
    return result


def build_catalog(settings: Settings) -> dict:
    """Costruisce il catalogo 2026 con codici provinciali ISTAT e punti espliciti."""
    collection = _get(PROVINCES_URL, timeout=settings.request_timeout_seconds).json()
    rows = list(csv.DictReader(io.StringIO(_get(DPC_URL, timeout=settings.request_timeout_seconds).text)))
    dpc = {row["codice_provincia"]: row for row in rows if row.get("lat") and row.get("long")}
    provinces = []
    regions: dict[str, dict] = {}
    points = {}
    for feature in collection["features"]:
        prop = feature["properties"]
        code = prop["prov_istat_code"]
        region = prop["reg_istat_code"]
        names = SPECIAL_CAPITALS.get(code)
        if names is None:
            row = dpc.get(code)
            if row is None:
                raise ValueError(f"Capoluogo mancante per la provincia {code}")
            names = [row["denominazione_provincia"]]
        point_ids = []
        for index, name in enumerate(names):
            point_id = f"p{code}-{index + 1}"
            if code not in SPECIAL_CAPITALS:
                latitude, longitude = float(dpc[code]["lat"]), float(dpc[code]["long"])
            else:
                payload = _get(GEOCODING_URL, params={"name": name, "count": 10, "language": "it", "format": "json"}, timeout=settings.request_timeout_seconds).json()
                matches = [
                    item for item in payload.get("results", [])
                    if item.get("country_code") == "IT" and item.get("name", "").casefold() == name.casefold()
                ]
                if not matches:
                    raise ValueError(f"Geocodifica non disponibile per {name}")
                match = max(matches, key=lambda item: item.get("population", 0))
                latitude, longitude = float(match["latitude"]), float(match["longitude"])
                time.sleep(0.2)
            points[point_id] = {"id": point_id, "name": name, "latitude": latitude, "longitude": longitude, "province_id": code}
            point_ids.append(point_id)
        provinces.append({"id": code, "name": prop["prov_name"], "region_id": region, "points": point_ids})
        regions[region] = {"id": region, "name": prop["reg_name"], "points": []}
    if len(provinces) != 110 or len(regions) != 20:
        raise ValueError(f"Catalogo incompleto: {len(provinces)} province, {len(regions)} regioni")
    for code, region in regions.items():
        province_id = REGION_CAPITAL_PROVINCE[code]
        province = next(item for item in provinces if item["id"] == province_id)
        region["points"] = [province["points"][0]]
    catalog = {
        "version": "2026.1", "points": points,
        "provinces": sorted(provinces, key=lambda item: item["name"]),
        "regions": sorted(regions.values(), key=lambda item: item["name"]),
        "sources": [PROVINCES_URL, DPC_URL, GEOCODING_URL],
    }
    directory = national_dir(settings)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    return catalog


def point_settings(settings: Settings, point: dict) -> Settings:
    return replace(
        settings,
        location_name=point["name"],
        latitude=point["latitude"],
        longitude=point["longitude"],
        data_dir=national_dir(settings) / "points" / point["id"],
    )


@contextmanager
def _budgeted_session(settings: Settings, daily_budget: int):
    if not 1 <= daily_budget <= 4000:
        raise ValueError("Il budget giornaliero deve essere compreso fra 1 e 4000 chiamate equivalenti.")
    root = national_dir(settings)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "fetch.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("Acquisizione nazionale già attiva o lock residuo: verificare prima di rimuoverlo.") from exc
    with os.fdopen(fd, "w", encoding="ascii") as file:
        file.write(str(os.getpid()))

    try:
        ledger_path = root / "request-budget.json"
        if ledger_path.is_file():
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            events = ledger["events"]
        else:
            events = []
            now = datetime.now(timezone.utc).timestamp()
            for path in (root / "points").glob("*/raw/archive_*.json"):
                modified = path.stat().st_mtime
                if 0 <= now - modified < 86400:
                    match = re.fullmatch(r"archive_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.json", path.name)
                    if match:
                        days = (date.fromisoformat(match[2]) - date.fromisoformat(match[1])).days + 1
                        events.append({"timestamp": modified, "cost": math.ceil(days / 14)})
            ledger_path.write_text(json.dumps({"events": events}), encoding="utf-8")

        class CachedSession(requests.Session):
            point_dir: Path

            def get(self, url, **kwargs):
                nonlocal events
                params = kwargs["params"]
                start, end = date.fromisoformat(params["start_date"]), date.fromisoformat(params["end_date"])
                raw = self.point_dir / "raw" / f"archive_{start}_{end}.json"
                if raw.is_file():
                    payload = json.loads(raw.read_text(encoding="utf-8"))
                    hourly = payload.get("hourly", {})
                    expected = pd.date_range(start.isoformat(), periods=((end - start).days + 1) * 24, freq="h", tz="UTC")
                    timestamps = pd.to_datetime(hourly.get("time", []), utc=True, errors="coerce")
                    if not timestamps.equals(expected) or any(len(hourly.get(name, [])) != len(expected) for name in HOURLY_VARIABLES):
                        raise ValueError(f"Blocco salvato non valido: {raw}")
                    response = requests.Response()
                    response.status_code = 200
                    response._content = json.dumps(payload).encode("utf-8")
                    return response
                now = datetime.now(timezone.utc).timestamp()
                events = [event for event in events if 0 <= now - event["timestamp"] < 86400]
                spent = sum(event["cost"] for event in events)
                hourly_spent = sum(event["cost"] for event in events if now - event["timestamp"] < 3600)
                cost = math.ceil(((end - start).days + 1) / 14)
                if spent + cost > daily_budget:
                    raise NationalBudgetExceeded(
                        f"Budget sulle 24 ore raggiunto ({spent}/{daily_budget} chiamate equivalenti). "
                        "Acquisizione incompleta: ripetere il comando dopo 24 ore."
                    )
                if hourly_spent + cost > 2000:
                    raise NationalBudgetExceeded(
                        f"Budget orario raggiunto ({hourly_spent}/2000 chiamate equivalenti). "
                        "Acquisizione incompleta: ripetere il comando dopo un'ora."
                    )
                events.append({"timestamp": now, "cost": cost})
                temporary = ledger_path.with_suffix(".tmp")
                temporary.write_text(json.dumps({"events": events}), encoding="utf-8")
                temporary.replace(ledger_path)
                time.sleep(1)
                return super().get(url, **kwargs)

        with CachedSession() as session:
            yield session
    finally:
        lock.unlink()


def verify_point_identity(local: Settings, point: dict) -> None:
    identity = {"latitude": point["latitude"], "longitude": point["longitude"], "name": point["name"]}
    identity_path = local.data_dir / "identity.json"
    if not identity_path.is_file():
        raise FileNotFoundError(f"Identità geografica non registrata per {point['name']}: riprendere national-fetch.")
    if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError(f"Coordinate del punto {point['id']} diverse dai dati salvati")


def _check_point_identity(local: Settings, point: dict) -> None:
    identity_path = local.data_dir / "identity.json"
    if not identity_path.is_file():
        raw_files = list(local.raw_dir.glob("archive_*.json"))
        if local.hourly_csv.is_file() and not raw_files:
            raise ValueError(f"Provenienza dello storico non verificabile per {point['name']}")
        for raw in raw_files:
            payload = json.loads(raw.read_text(encoding="utf-8"))
            if abs(payload["latitude"] - point["latitude"]) > .2 or abs(payload["longitude"] - point["longitude"]) > .2:
                raise ValueError(f"Coordinate incompatibili nel blocco {raw}")
        local.data_dir.mkdir(parents=True, exist_ok=True)
        identity = {"latitude": point["latitude"], "longitude": point["longitude"], "name": point["name"]}
        identity_path.write_text(json.dumps(identity, ensure_ascii=False), encoding="utf-8")
    verify_point_identity(local, point)


def fetch_national_history(settings: Settings, catalog: dict, daily_budget: int = 4000) -> None:
    root = national_dir(settings)
    with _budgeted_session(settings, daily_budget) as session:
        today = datetime.now(timezone.utc).date()
        acquisition_path = root / "acquisition.json"
        fingerprint = hashlib.sha256(json.dumps(catalog, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        if acquisition_path.is_file():
            acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
            if acquisition["catalog_sha256"] != fingerprint or acquisition["history_years"] != settings.history_years:
                raise ValueError("Il catalogo o history_years è cambiato rispetto all'acquisizione nazionale in corso.")
            end_date = date.fromisoformat(acquisition["end_date"])
        else:
            end_date = today - timedelta(days=1)
            acquisition = {"catalog_sha256": fingerprint, "history_years": settings.history_years, "end_date": end_date.isoformat()}
            acquisition_path.write_text(json.dumps(acquisition, ensure_ascii=False, indent=2), encoding="utf-8")
        for point in catalog["points"].values():
            local = point_settings(settings, point)
            _check_point_identity(local, point)
            if local.hourly_csv.is_file():
                read_hourly_csv(local.hourly_csv)
                LOGGER.info("Storico già presente per %s, ignorato", point["name"])
                continue
            LOGGER.info("Download storico per %s (%s)", point["name"], point["id"])
            session.point_dir = local.data_dir
            fetch_history(local, session=session, end_date=end_date)


def refresh_national_history(settings: Settings, catalog: dict, daily_budget: int = 4000) -> None:
    """Aggiunge solo le giornate successive all'ultima già archiviata."""
    end_date = datetime.now(timezone.utc).date() - timedelta(days=1)
    with _budgeted_session(settings, daily_budget) as session:
        for point in catalog["points"].values():
            local = point_settings(settings, point)
            _check_point_identity(local, point)
            session.point_dir = local.data_dir
            existing = read_hourly_csv(local.hourly_csv)
            last_day = existing["timestamp"].max().date()
            last_count = (existing["timestamp"].dt.date == last_day).sum()
            if last_count != 24:
                raise ValueError(f"Ultima giornata incompleta per {point['name']}")
            cursor = last_day + timedelta(days=1)
            frames = [existing]
            while cursor <= end_date:
                chunk_end = min(cursor + timedelta(days=89), end_date)
                params = {
                    "latitude": point["latitude"], "longitude": point["longitude"],
                    "start_date": cursor.isoformat(), "end_date": chunk_end.isoformat(),
                    "hourly": ",".join(HOURLY_VARIABLES), "timezone": "UTC",
                    "temperature_unit": "celsius", "wind_speed_unit": "kmh", "precipitation_unit": "mm",
                }
                try:
                    response = session.get(ARCHIVE_URL, params=params, timeout=settings.request_timeout_seconds)
                    response.raise_for_status()
                    payload = response.json()
                except (requests.RequestException, requests.JSONDecodeError) as exc:
                    raise RuntimeError(f"Aggiornamento {point['name']} {cursor}–{chunk_end} fallito: {exc}") from exc
                if not isinstance(payload, dict) or "error" in payload:
                    raise RuntimeError(f"Risposta Open-Meteo non valida per {point['name']}: {payload}")
                hourly = payload.get("hourly")
                if not isinstance(hourly, dict) or any(key not in hourly for key in ["time", *HOURLY_VARIABLES]):
                    raise RuntimeError(f"Risposta oraria incompleta per {point['name']}")
                expected = pd.date_range(
                    cursor.isoformat(), periods=((chunk_end - cursor).days + 1) * 24, freq="h", tz="UTC"
                )
                timestamps = pd.to_datetime(hourly["time"], utc=True, errors="coerce")
                if not timestamps.equals(expected) or any(len(hourly[key]) != len(expected) for key in HOURLY_VARIABLES):
                    raise RuntimeError(f"Serie oraria incompleta o non allineata per {point['name']}")
                frame = pd.DataFrame({key: hourly[key] for key in HOURLY_VARIABLES})
                frame.insert(0, "timestamp", timestamps)
                frame["location"] = point["name"]
                frames.append(frame)
                local.raw_dir.mkdir(parents=True, exist_ok=True)
                (local.raw_dir / f"archive_{cursor}_{chunk_end}.json").write_text(json.dumps(payload), encoding="utf-8")
                cursor = chunk_end + timedelta(days=1)
            if len(frames) > 1:
                updated = validate_hourly_data(pd.concat(frames, ignore_index=True))
                temporary = local.hourly_csv.with_suffix(".tmp")
                updated.to_csv(temporary, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
                temporary.replace(local.hourly_csv)
                LOGGER.info("Storico aggiornato per %s fino al %s", point["name"], end_date)
