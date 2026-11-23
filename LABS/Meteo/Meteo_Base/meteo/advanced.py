"""Backtest rolling per temperatura e vento con previsioni NWP archiviate."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from meteo.config import Settings
from meteo.data import validate_hourly_data
from meteo.experiments import _add_recent_features
from meteo.features import aggregate_daily, build_training_frame
from meteo.nwp_history import NWP_VARIABLES, read_nwp_history
from meteo.training import _regression_metrics, _save_test_plot

LOGGER = logging.getLogger(__name__)
TARGETS = ("temperature_2m_max", "wind_speed_10m_max")
UNITS = {"temperature_2m_max": "°C", "wind_speed_10m_max": "km/h"}
NWP_BASELINES = {
    "temperature_2m_max": "nwp_temperature_2m_max",
    "wind_speed_10m_max": "nwp_wind_speed_10m_max",
}


@dataclass
class ResidualModel:
    estimator: Any
    reference_column: str

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "ResidualModel":
        self.estimator.fit(x, y - x[self.reference_column])
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return x[self.reference_column].to_numpy() + self.estimator.predict(x)


def _hourly_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Costruisce feature giornaliere usando solo ore fino al termine di D."""
    data = validate_hourly_data(hourly)
    needed = ["temperature_2m", "pressure_msl", "wind_speed_10m", "wind_direction_10m"]
    if data[needed].isna().any().any():
        raise ValueError("Le feature avanzate richiedono dati orari completi per temperatura, pressione e vento")
    data["date"] = data["timestamp"].dt.floor("D")
    grouped = data.groupby("date", sort=True)
    direction = np.deg2rad(data["wind_direction_10m"])
    data["wind_direction_sin"] = np.sin(direction)
    data["wind_direction_cos"] = np.cos(direction)
    result = grouped[["temperature_2m", "pressure_msl", "wind_speed_10m"]].std().rename(
        columns={
            "temperature_2m": "temperature_hourly_std",
            "pressure_msl": "pressure_hourly_std",
            "wind_speed_10m": "wind_hourly_std",
        }
    )
    result["wind_direction_sin_mean"] = grouped["wind_direction_sin"].mean()
    result["wind_direction_cos_mean"] = grouped["wind_direction_cos"].mean()
    first = data.loc[data["timestamp"].dt.hour < 6].groupby("date")[["temperature_2m", "pressure_msl"]].mean()
    last = data.loc[data["timestamp"].dt.hour >= 18].groupby("date")[["temperature_2m", "pressure_msl"]].mean()
    result["temperature_day_trend"] = last["temperature_2m"] - first["temperature_2m"]
    result["pressure_day_trend"] = last["pressure_msl"] - first["pressure_msl"]
    return result.reset_index()


def _build_frame(
    hourly: pd.DataFrame, nwp: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    daily = aggregate_daily(hourly)
    if daily["date"].diff().dropna().ne(pd.Timedelta(days=1)).any():
        raise ValueError("Lo storico contiene giorni incompleti: impossibile allineare i target D+2")
    frame, base_columns = build_training_frame(hourly)
    extra = _hourly_features(hourly)
    extra_columns = [column for column in extra.columns if column != "date"]
    frame = frame.merge(extra, on="date", how="left", validate="one_to_one")
    if frame[extra_columns].isna().any().any():
        raise ValueError("Feature orarie avanzate incomplete")
    recent = _add_recent_features(frame, base_columns)
    added = [name for name in recent.columns if name not in base_columns]
    frame[added] = recent[added]
    frame["target_date"] = frame["date"] + pd.Timedelta(days=2)
    original_rows = len(frame)
    frame = frame.merge(nwp, on="target_date", how="inner", validate="one_to_one")
    if frame.empty:
        raise ValueError("Nessuna data comune tra storico e previsioni NWP")
    # L'anticipo nominale deve precedere la fine del giorno delle feature.
    if (frame["latest_nominal_forecast_time"] > frame["date"] + pd.Timedelta(hours=23)).any():
        raise ValueError("La previsione NWP nominale è successiva al momento della previsione")
    nwp_columns = list(NWP_VARIABLES.values())
    columns = [*base_columns, *added, *extra_columns, *nwp_columns]
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError("Le feature contengono valori non finiti")
    LOGGER.info("Allineate %d giornate su %d; NWP non disponibile per %d",
                len(frame), original_rows, original_rows - len(frame))
    return frame.sort_values("date").reset_index(drop=True), columns, {
        "history_examples": original_rows,
        "nwp_aligned_examples": len(frame),
        "excluded_without_nwp": original_rows - len(frame),
    }


def _candidates(target: str) -> dict[str, Any]:
    candidates: dict[str, Any] = {
        "random_forest": RandomForestRegressor(
            n_estimators=100, min_samples_leaf=6, max_features=0.8, random_state=42, n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=100, min_samples_leaf=4, max_features=0.8, random_state=42, n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=150, max_leaf_nodes=15, l2_regularization=1.0, early_stopping=False, random_state=42,
        ),
    }
    if target == "temperature_2m_max":
        candidates["hist_boost_residual_persistence"] = ResidualModel(
            HistGradientBoostingRegressor(
                max_iter=150, max_leaf_nodes=15, l2_regularization=1.0,
                early_stopping=False, random_state=42,
            ),
            "temperature_2m_max",
        )
    return candidates


def _new_model(model: Any) -> Any:
    if isinstance(model, ResidualModel):
        return ResidualModel(clone(model.estimator), model.reference_column)
    return clone(model)


def _groups(
    actual: np.ndarray, predicted: np.ndarray, baseline: np.ndarray, labels: pd.Series,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for label in sorted(labels.dropna().unique()):
        mask = labels.to_numpy() == label
        result[str(label)] = {
            "samples": int(mask.sum()),
            "mae_model": float(mean_absolute_error(actual[mask], predicted[mask])),
            "mae_persistence": float(mean_absolute_error(actual[mask], baseline[mask])),
        }
    return result


def run_advanced_experiment(hourly: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    """Seleziona candidati con tre finestre rolling prima del test finale."""
    nwp = read_nwp_history(settings)
    frame, feature_columns, coverage = _build_frame(hourly, nwp)
    if len(frame) < 150:
        raise ValueError("Servono almeno 150 giorni allineati tra storico e NWP")
    test_start_index = int(len(frame) * 0.85)
    test = frame.iloc[test_start_index:]
    pretest = frame.iloc[:test_start_index]
    pretest = pretest.loc[pretest["target_date"] <= test["date"].min()].reset_index(drop=True)
    if len(pretest) < 100:
        raise ValueError("Dati pre-test insufficienti dopo la separazione temporale")

    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for train_index, validation_index in TimeSeriesSplit(n_splits=3).split(pretest):
        train = pretest.iloc[train_index]
        validation = pretest.iloc[validation_index]
        train = train.loc[train["target_date"] <= validation["date"].min()]
        if len(train) < 30:
            raise ValueError("Una finestra rolling non ha abbastanza esempi di training")
        folds.append((train, validation))

    report: dict[str, Any] = {
        "coverage": coverage,
        "selection": "MAE aggregato sui tre fold rolling pre-test",
        "nwp_availability_warning": (
            "Previous Runs usa un anticipo nominale di 72 ore; non espone l'orario "
            "effettivo di pubblicazione. La disponibilità alla previsione non è certificata."
        ),
        "split": {
            "pretest_samples": len(pretest), "test_samples": len(test),
            "pretest_last_target_date": pretest["target_date"].max().isoformat(),
            "test_first_feature_date": test["date"].min().isoformat(),
            "test_first_target_date": test["target_date"].iloc[0].isoformat(),
            "test_last_target_date": test["target_date"].iloc[-1].isoformat(),
            "folds": [
                {
                    "train_samples": len(train), "validation_samples": len(validation),
                    "train_last_target_date": train["target_date"].max().isoformat(),
                    "validation_first_feature_date": validation["date"].min().isoformat(),
                    "validation_last_feature_date": validation["date"].max().isoformat(),
                }
                for train, validation in folds
            ],
        },
        "targets": {},
        "limitations": [
            "Dataset NWP disponibile solo per parte dello storico; i risultati non sono direttamente confrontabili con esperimenti su periodi diversi.",
            "Rianalisi Open-Meteo usata come riferimento, non osservazioni indipendenti di stazione.",
            "L'archivio Previous Runs non documenta la disponibilità effettiva di ogni singolo dato al momento della previsione.",
            "I modelli sperimentali non modificano l'API operativa e non sono esportati in ONNX.",
        ],
    }
    output = settings.artifacts_dir / "advanced_experiments"
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    selected_models: dict[str, Any] = {}
    for target in TARGETS:
        candidates = _candidates(target)
        candidate_scores: dict[str, Any] = {}
        for name, template in candidates.items():
            fold_scores = []
            total_error = 0.0
            total_count = 0
            for train, validation in folds:
                model = _new_model(template)
                model.fit(train[feature_columns], train[f"target_{target}"])
                actual = validation[f"target_{target}"].to_numpy(dtype=float)
                prediction = model.predict(validation[feature_columns])
                error = float(mean_absolute_error(actual, prediction))
                fold_scores.append({"mae": error, "samples": len(validation)})
                total_error += error * len(validation)
                total_count += len(validation)
            candidate_scores[name] = {"mae": total_error / total_count, "folds": fold_scores}

        selected_name = min(candidate_scores, key=lambda name: candidate_scores[name]["mae"])
        selected = _new_model(candidates[selected_name])
        selected.fit(pretest[feature_columns], pretest[f"target_{target}"])
        selected_models[target] = selected
        y = test[f"target_{target}"].to_numpy(dtype=float)
        prediction = selected.predict(test[feature_columns])
        persistence = test[target].to_numpy(dtype=float)
        nwp_prediction = test[NWP_BASELINES[target]].to_numpy(dtype=float)
        month = test["target_date"].dt.month
        seasons = month.map({12: "inverno", 1: "inverno", 2: "inverno", 3: "primavera",
                             4: "primavera", 5: "primavera", 6: "estate", 7: "estate",
                             8: "estate", 9: "autunno", 10: "autunno", 11: "autunno"})
        result: dict[str, Any] = {
            "unit": UNITS[target], "selected": selected_name,
            "rolling_candidates": candidate_scores,
            "test_selected": _regression_metrics(y, prediction),
            "test_persistence": _regression_metrics(y, persistence),
            "test_nwp": _regression_metrics(y, nwp_prediction),
            "test_by_season": _groups(y, prediction, persistence, seasons),
        }
        if target == "wind_speed_10m_max":
            cutoffs = pretest[f"target_{target}"].quantile([1 / 3, 2 / 3]).to_numpy()
            wind_labels = pd.Series(
                np.where(y <= cutoffs[0], "basso", np.where(y <= cutoffs[1], "medio", "alto"))
            )
            result["wind_intensity_cutoffs_kmh"] = cutoffs.tolist()
            result["test_by_observed_wind_intensity"] = _groups(y, prediction, persistence, wind_labels)
        report["targets"][target] = result
        _save_test_plot(plots / f"{target}_test.png", target, UNITS[target],
                        test["date"], y, prediction, persistence)

    joblib.dump({"models": selected_models, "feature_columns": feature_columns},
                output / "models.joblib")
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Risultati avanzati salvati in %s", output)
    return report


def format_advanced_results(report: dict[str, Any]) -> str:
    lines = [
        "Confronto sul test finale: selezione con 3 finestre rolling",
        f"{'Target':<20} {'Modello':<30} {'MAE ML':>8} {'MAE pers.':>10} {'MAE NWP':>8} {'R2 ML':>8}",
    ]
    for target, result in report["targets"].items():
        lines.append(
            f"{target:<20} {result['selected']:<30} "
            f"{result['test_selected']['mae']:>8.3f} {result['test_persistence']['mae']:>10.3f} "
            f"{result['test_nwp']['mae']:>8.3f} {result['test_selected']['r2']:>8.3f}"
        )
    lines.append("NWP: anticipo nominale 72 ore, disponibilita effettiva non certificata.")
    return "\n".join(lines)
