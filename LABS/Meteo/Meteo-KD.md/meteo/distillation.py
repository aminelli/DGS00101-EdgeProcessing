"""Distillazione di regressori territoriali da un teacher nazionale."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from meteo.config import Settings
from meteo.data import DAILY_TARGETS, HOURLY_VARIABLES, read_hourly_csv
from meteo.features import aggregate_daily, build_training_frame, latest_feature_row
from meteo.inference import UNITS
from meteo.national import active_distillation_dir, load_catalog, point_settings, verify_point_identity
from meteo.training import _regression_metrics

DISTILLATION_WEIGHT = 0.7
GEOGRAPHIC_FEATURES = ["site_latitude", "site_longitude"]


def _physical_predictions(target: str, values: np.ndarray) -> np.ndarray:
    if target in ("precipitation_sum", "wind_speed_10m_max"):
        return np.maximum(values, 0)
    return values


def _point_frame(hourly: pd.DataFrame, point: dict) -> tuple[pd.DataFrame, list[str]]:
    _validate_hourly(hourly, point)
    daily = aggregate_daily(hourly)
    if daily.empty or not daily["date"].diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError(f"Giornate mancanti o non consecutive per {point['name']}")
    frame, columns = build_training_frame(hourly)
    frame["site_latitude"] = point["latitude"]
    frame["site_longitude"] = point["longitude"]
    columns = [*columns, *GEOGRAPHIC_FEATURES]
    required = [*columns, *(f"target_{target}" for target in DAILY_TARGETS)]
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError(f"Feature o target non finiti per {point['name']}")
    return frame, columns


def _validate_hourly(hourly: pd.DataFrame, point: dict) -> None:
    if not hourly["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError(f"Giornate mancanti o non consecutive per {point['name']}")
    if not np.isfinite(hourly[HOURLY_VARIABLES].to_numpy(dtype=float)).all():
        raise ValueError(f"Valori orari mancanti o non finiti per {point['name']}")
    if not hourly["location"].eq(point["name"]).all():
        raise ValueError(f"La località nei dati non coincide con {point['name']}")


def train_distillation(settings: Settings, catalog: dict | None = None) -> dict[str, Any]:
    catalog = load_catalog(settings) if catalog is None else catalog
    missing = [
        point["name"] for point in catalog["points"].values()
        if not point_settings(settings, point).hourly_csv.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Storici nazionali mancanti: {len(missing)} punti; "
            f"ad esempio {', '.join(missing[:5])}. Riprendere national-fetch."
        )
    frames: dict[str, pd.DataFrame] = {}
    feature_columns: list[str] | None = None
    common_dates: set[pd.Timestamp] | None = None
    for point_id, point in catalog["points"].items():
        local = point_settings(settings, point)
        verify_point_identity(local, point)
        frame, columns = _point_frame(read_hourly_csv(local.hourly_csv), point)
        if feature_columns is not None and columns != feature_columns:
            raise ValueError("Le feature dei punti nazionali non coincidono")
        feature_columns = columns
        dates = set(frame["date"])
        common_dates = dates if common_dates is None else common_dates & dates
        frames[point_id] = frame
    if not feature_columns or common_dates is None:
        raise ValueError("Il catalogo non contiene punti per il training")
    dates = sorted(common_dates)
    if len(dates) < 120:
        raise ValueError(f"Servono almeno 120 giornate valide comuni; disponibili {len(dates)}")
    validation_start = dates[int(len(dates) * .70)]
    test_start = dates[int(len(dates) * .85)]
    for point_id, frame in frames.items():
        frames[point_id] = frame[frame["date"].isin(common_dates)].reset_index(drop=True)

    def split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = frame[frame["date"] + pd.Timedelta(days=2) < validation_start]
        validation = frame[(frame["date"] >= validation_start) & (frame["date"] + pd.Timedelta(days=2) < test_start)]
        test = frame[frame["date"] >= test_start]
        if min(len(train), len(validation), len(test)) < 2:
            raise ValueError("Split temporale insufficiente per la distillazione")
        return train, validation, test

    parts = {point_id: split(frame) for point_id, frame in frames.items()}
    national_train = pd.concat([item[0] for item in parts.values()], ignore_index=True)
    teachers = {}
    for target in DAILY_TARGETS:
        teacher = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=31, early_stopping=False, random_state=42)
        teacher.fit(national_train[feature_columns], national_train[f"target_{target}"])
        teachers[target] = teacher

    root = settings.artifacts_dir / "distillation"
    run_id = uuid4().hex
    directory = root / "runs" / run_id
    student_dir = directory / "students"
    student_dir.mkdir(parents=True)
    teacher_path = directory / "teacher.joblib"
    joblib.dump({"models": teachers, "feature_columns": feature_columns}, teacher_path)
    metrics: dict[str, Any] = {
        "distillation_weight": DISTILLATION_WEIGHT,
        "teacher_size_bytes": teacher_path.stat().st_size,
        "split": {
            "validation_first_feature_date": validation_start.isoformat(),
            "test_first_feature_date": test_start.isoformat(),
            "test_last_feature_date": dates[-1].isoformat(),
        },
        "territories": {},
        "limitations": [
            "Previsioni puntuali nei capoluoghi, non medie territoriali.",
            "Teacher e student sono valutati su giornate future rispetto al training.",
            "Rianalisi Open-Meteo, non osservazioni indipendenti di stazioni.",
        ],
    }
    for level, items in (("region", catalog["regions"]), ("province", catalog["provinces"])):
        for territory in items:
            key = f"{level}:{territory['id']}"
            train = pd.concat([parts[p][0] for p in territory["points"]], ignore_index=True)
            test = pd.concat([parts[p][2] for p in territory["points"]], ignore_index=True)
            models = {}
            results = {}
            for target in DAILY_TARGETS:
                teacher = teachers[target]
                train_x = train[feature_columns]
                teacher_train = _physical_predictions(target, teacher.predict(train_x))
                labels = train[f"target_{target}"].to_numpy(dtype=float)
                soft_labels = DISTILLATION_WEIGHT * teacher_train + (1 - DISTILLATION_WEIGHT) * labels
                student = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                student.fit(train_x, soft_labels)
                supervised = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                supervised.fit(train_x, labels)
                test_x = test[feature_columns]
                truth = test[f"target_{target}"].to_numpy(dtype=float)
                results[target] = {
                    "unit": UNITS[target],
                    "student": _regression_metrics(truth, _physical_predictions(target, student.predict(test_x))),
                    "supervised_student": _regression_metrics(truth, _physical_predictions(target, supervised.predict(test_x))),
                    "teacher": _regression_metrics(truth, _physical_predictions(target, teacher.predict(test_x))),
                    "persistence": _regression_metrics(truth, test[target].to_numpy(dtype=float)),
                }
                models[target] = student
            student_path = student_dir / f"{level}-{territory['id']}.joblib"
            joblib.dump({"models": models, "feature_columns": feature_columns}, student_path)
            metrics["territories"][key] = {
                "name": territory["name"], "points": territory["points"],
                "test_samples": len(test), "student_size_bytes": student_path.stat().st_size, "targets": results,
            }
    (directory / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (directory / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    pointer = root / f"current-{run_id}.tmp"
    pointer.write_text(run_id, encoding="ascii")
    pointer.replace(root / "current")
    return metrics


def forecast_student(settings: Settings, level: str, territory_id: str, point_id: str) -> dict[str, Any]:
    if level not in ("region", "province"):
        raise ValueError("Livello territoriale non valido")
    directory = active_distillation_dir(settings)
    catalog_path = directory / "catalog.json"
    if not catalog_path.is_file():
        raise FileNotFoundError("Modelli nazionali non disponibili: eseguire national-train.")
    catalog = load_catalog(settings, catalog_path)
    territory = next((item for item in catalog[level + "s"] if item["id"] == territory_id), None)
    if territory is None or point_id not in territory["points"]:
        raise ValueError("Territorio o capoluogo non valido")
    path = directory / "students" / f"{level}-{territory_id}.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"Student non disponibile per {territory['name']}.")
    artifact = joblib.load(path)
    point = catalog["points"][point_id]
    local = point_settings(settings, point)
    verify_point_identity(local, point)
    hourly = read_hourly_csv(local.hourly_csv)
    _validate_hourly(hourly, point)
    local_columns = [column for column in artifact["feature_columns"] if column not in GEOGRAPHIC_FEATURES]
    features = latest_feature_row(hourly, local_columns)
    features["site_latitude"] = point["latitude"]
    features["site_longitude"] = point["longitude"]
    latest = aggregate_daily(hourly)["date"].max().date()
    model = artifact["models"]
    return {
        "territory": territory["name"], "level": level, "point": point,
        "latest_history_date": latest.isoformat(),
        "forecast_date": (latest + timedelta(days=2)).isoformat(),
        "units": UNITS,
        "student": {
            target: float(_physical_predictions(target, model[target].predict(features[artifact["feature_columns"]]))[0])
            for target in DAILY_TARGETS
        },
        "disclaimer": "Previsione puntuale nel capoluogo, non rappresentativa dell'intero territorio.",
    }
