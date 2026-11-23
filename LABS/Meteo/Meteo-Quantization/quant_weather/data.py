"""Caricamento e validazione del dataset prodotto dalla pipeline meteo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMNS = (
    "target_temperature_2m_max",
    "target_precipitation_sum",
    "target_wind_speed_10m_max",
)


@dataclass(frozen=True)
class Dataset:
    features: pd.DataFrame
    targets: pd.DataFrame
    persistence: pd.DataFrame
    dates: pd.Series


def load_dataset(path: str | Path) -> Dataset:
    """Legge training_features.csv senza modificare il file sorgente."""
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset non trovato: {csv_path}")

    frame = pd.read_csv(csv_path)
    if "date" not in frame.columns:
        raise ValueError("Il dataset deve contenere la colonna 'date'")
    missing_targets = set(TARGET_COLUMNS) - set(frame.columns)
    if missing_targets:
        raise ValueError(f"Target mancanti nel dataset: {sorted(missing_targets)}")

    dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    if dates.isna().any():
        raise ValueError("Il dataset contiene date non valide")
    if dates.duplicated().any():
        raise ValueError("Il dataset contiene date duplicate")

    persistence_columns = {
        target: target.removeprefix("target_") for target in TARGET_COLUMNS
    }
    missing_persistence = set(persistence_columns.values()) - set(frame.columns)
    if missing_persistence:
        raise ValueError(
            f"Colonne baseline mancanti nel dataset: {sorted(missing_persistence)}"
        )

    excluded = {"date", *TARGET_COLUMNS}
    feature_frame = frame.drop(columns=list(excluded))
    non_numeric = [
        column
        for column in feature_frame.columns
        if not pd.api.types.is_numeric_dtype(feature_frame[column])
    ]
    if non_numeric:
        raise ValueError(f"Feature non numeriche non supportate: {non_numeric}")

    target_frame = frame[list(TARGET_COLUMNS)].astype(np.float32)
    persistence_frame = frame[list(persistence_columns.values())].astype(np.float32)
    numeric_features = feature_frame.astype(np.float32)
    if not np.isfinite(numeric_features.to_numpy()).all():
        raise ValueError("Le feature contengono valori mancanti o non finiti")
    if not np.isfinite(target_frame.to_numpy()).all():
        raise ValueError("I target contengono valori mancanti o non finiti")
    if not np.isfinite(persistence_frame.to_numpy()).all():
        raise ValueError("La baseline contiene valori mancanti o non finiti")

    order = np.argsort(dates.to_numpy())
    return Dataset(
        features=numeric_features.iloc[order].reset_index(drop=True),
        targets=target_frame.iloc[order].reset_index(drop=True),
        persistence=persistence_frame.iloc[order].reset_index(drop=True),
        dates=dates.iloc[order].reset_index(drop=True),
    )
