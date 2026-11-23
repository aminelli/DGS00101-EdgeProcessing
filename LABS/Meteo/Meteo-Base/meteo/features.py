"""Aggregazione giornaliera e feature causali per la previsione a 24 ore."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from meteo.data import DAILY_TARGETS, HOURLY_VARIABLES, validate_hourly_data

LOGGER = logging.getLogger(__name__)

INPUT_DAILY_COLUMNS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "dew_point_2m_mean",
    "precipitation_sum",
    "pressure_msl_mean",
    "cloud_cover_mean",
    "wind_speed_10m_mean",
    "wind_speed_10m_max",
    "shortwave_radiation_sum",
]
TARGET_COLUMNS = DAILY_TARGETS


def aggregate_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Calcola riepiloghi UTC giornalieri e conserva solo giorni completi."""
    checked = validate_hourly_data(hourly)
    checked["date"] = checked["timestamp"].dt.floor("D")
    grouped = checked.groupby("date", sort=True)
    daily = pd.DataFrame(index=grouped.size().index)
    daily["temperature_2m_mean"] = grouped["temperature_2m"].mean()
    daily["temperature_2m_max"] = grouped["temperature_2m"].max()
    daily["temperature_2m_min"] = grouped["temperature_2m"].min()
    daily["relative_humidity_2m_mean"] = grouped["relative_humidity_2m"].mean()
    daily["dew_point_2m_mean"] = grouped["dew_point_2m"].mean()
    daily["precipitation_sum"] = grouped["precipitation"].sum(min_count=1)
    daily["pressure_msl_mean"] = grouped["pressure_msl"].mean()
    daily["cloud_cover_mean"] = grouped["cloud_cover"].mean()
    daily["wind_speed_10m_mean"] = grouped["wind_speed_10m"].mean()
    daily["wind_speed_10m_max"] = grouped["wind_speed_10m"].max()
    daily["shortwave_radiation_sum"] = grouped["shortwave_radiation"].sum(min_count=1)
    daily["hour_count"] = grouped["timestamp"].count()
    incomplete_days = int((daily["hour_count"] != 24).sum())
    if incomplete_days:
        LOGGER.warning("Escluse %d giornate con copertura oraria incompleta", incomplete_days)
    daily = daily.loc[daily["hour_count"] == 24].drop(columns="hour_count")
    return daily.reset_index()


def build_training_frame(hourly: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Crea esempi D→D+2, prevedendo domani da dati completi fino a ieri."""
    daily = aggregate_daily(hourly).sort_values("date").reset_index(drop=True)
    feature_columns: list[str] = []
    for column in INPUT_DAILY_COLUMNS:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
        feature_columns.append(column)
        for lag in (1, 2, 3, 7):
            name = f"{column}_lag_{lag}"
            daily[name] = daily[column].shift(lag)
            feature_columns.append(name)

    day_of_year = daily["date"].dt.day_of_year
    daily["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    daily["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    feature_columns.extend(["day_of_year_sin", "day_of_year_cos"])

    for column in TARGET_COLUMNS:
        daily[f"target_{column}"] = daily[column].shift(-2)
    targets = [f"target_{column}" for column in TARGET_COLUMNS]
    valid = daily.dropna(subset=feature_columns + targets)
    dropped = len(daily) - len(valid)
    if dropped:
        LOGGER.warning("Esclusi %d esempi con feature o target mancanti", dropped)
    return valid.reset_index(drop=True), feature_columns


def latest_feature_row(hourly: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Restituisce le feature dell'ultimo giorno completo con lo schema del training."""
    daily = aggregate_daily(hourly).sort_values("date").reset_index(drop=True)
    if len(daily) < 8:
        raise ValueError("Servono almeno 8 giorni completi per calcolare i lag configurati")
    row: dict[str, float] = {}
    for column in INPUT_DAILY_COLUMNS:
        values = daily[column].astype(float)
        row[column] = float(values.iloc[-1])
        for lag in (1, 2, 3, 7):
            row[f"{column}_lag_{lag}"] = float(values.iloc[-1 - lag])
    day_of_year = daily["date"].iloc[-1].day_of_year
    row["day_of_year_sin"] = float(np.sin(2 * np.pi * day_of_year / 365.25))
    row["day_of_year_cos"] = float(np.cos(2 * np.pi * day_of_year / 365.25))
    result = pd.DataFrame([row], columns=feature_columns)
    if result.isna().any().any():
        raise ValueError("Le feature dell'ultima giornata contengono valori mancanti")
    return result


def prepare_hourly_frame(hourly: pd.DataFrame) -> pd.DataFrame:
    """Verifica uno schema orario e scarta colonne estranee al modello."""
    checked = validate_hourly_data(hourly)
    return checked[["timestamp", "location", *HOURLY_VARIABLES]]
