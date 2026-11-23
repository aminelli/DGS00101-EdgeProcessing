"""Predizione ML e composizione trasparente con il dato NWP operativo."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from meteo.config import Settings
from meteo.data import DAILY_TARGETS, fetch_nwp_forecast, read_hourly_csv
from meteo.features import aggregate_daily, latest_feature_row

UNITS = {
    "temperature_2m_max": "°C",
    "precipitation_sum": "mm",
    "wind_speed_10m_max": "km/h",
}


def load_model(path: str | Path) -> dict[str, Any]:
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Modello non trovato: {model_path}. Eseguire prima fetch-history, prepare e train."
        )
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or not {"models", "feature_columns", "location"} <= set(artifact):
        raise ValueError(f"Artefatto modello non valido: {model_path}")
    return artifact


def make_forecast(settings: Settings) -> dict[str, Any]:
    """Prevede domani usando l'ultimo giorno UTC completo disponibile."""
    artifact = load_model(settings.model_path)
    location = artifact["location"]
    if (
        abs(float(location["latitude"]) - settings.latitude) > 1e-5
        or abs(float(location["longitude"]) - settings.longitude) > 1e-5
    ):
        raise ValueError("Il modello disponibile è addestrato per coordinate diverse da quelle configurate")
    hourly = read_hourly_csv(settings.hourly_csv)
    feature_row = latest_feature_row(hourly, artifact["feature_columns"])
    nwp = fetch_nwp_forecast(settings)
    latest_complete_date = aggregate_daily(hourly)["date"].max().date()
    forecast_date = latest_complete_date + timedelta(days=2)
    nwp_next = nwp.loc[nwp["date"].dt.date == forecast_date]
    ml_values: dict[str, float] = {}
    for target in DAILY_TARGETS:
        ml_values[target] = float(artifact["models"][target].predict(feature_row)[0])

    nwp_values: dict[str, float] | None = None
    hybrid_values: dict[str, float] | None = None
    if not nwp_next.empty:
        nwp_values = {target: float(nwp_next.iloc[0][target]) for target in DAILY_TARGETS}
        weight = settings.hybrid_ml_weight
        hybrid_values = {
            target: weight * ml_values[target] + (1 - weight) * nwp_values[target]
            for target in DAILY_TARGETS
        }

    return {
        "location": location,
        "forecast_date": forecast_date.isoformat(),
        "units": UNITS,
        "machine_learning": ml_values,
        "nwp": nwp.to_dict(orient="records"),
        "nwp_next_day": nwp_values,
        "hybrid_next_day": hybrid_values,
        "hybrid_ml_weight": settings.hybrid_ml_weight,
        "hybrid_status": (
            "Blend configurabile non calibrato: non è disponibile un archivio NWP storico "
            "allineato per una valutazione retrospettiva."
            if hybrid_values is not None
            else "Dato NWP non disponibile per la data target; il blend non è stato calcolato."
        ),
    }
