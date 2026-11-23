"""Confronto sperimentale di modelli, senza modificare gli artefatti operativi."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import brier_score_loss, confusion_matrix, mean_absolute_error

from meteo.config import Settings
from meteo.features import TARGET_COLUMNS, aggregate_daily, build_training_frame
from meteo.training import _regression_metrics, _save_test_plot

LOGGER = logging.getLogger(__name__)
RAIN_THRESHOLD_MM = 0.0
RAIN_PROBABILITY_THRESHOLD = 0.5
UNITS = {
    "temperature_2m_max": "°C",
    "precipitation_sum": "mm",
    "wind_speed_10m_max": "km/h",
}


def _forest(leaf: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=100, min_samples_leaf=leaf, max_features=0.8,
        random_state=42, n_jobs=-1,
    )


@dataclass
class ResidualTemperatureModel:
    estimator: RandomForestRegressor

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        self.estimator.fit(x, y - x["temperature_2m_max"])

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return x["temperature_2m_max"].to_numpy() + self.estimator.predict(x)


@dataclass
class TwoStageRainModel:
    classifier: RandomForestClassifier
    amount_model: RandomForestRegressor

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        rainy = y > RAIN_THRESHOLD_MM
        if rainy.nunique() != 2:
            raise ValueError("Il training pioggia richiede giorni asciutti e piovosi")
        self.classifier.fit(x, rainy)
        self.amount_model.fit(x.loc[rainy], y.loc[rainy])

    def predict_probability(self, x: pd.DataFrame) -> np.ndarray:
        positive_index = list(self.classifier.classes_).index(True)
        return self.classifier.predict_proba(x)[:, positive_index]

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.predict_probability(x) * self.amount_model.predict(x)


def _add_recent_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    features = frame[columns].copy()
    for column in ("temperature_2m_max", "precipitation_sum", "wind_speed_10m_max"):
        features[f"{column}_mean_3d"] = frame[
            [column, f"{column}_lag_1", f"{column}_lag_2"]
        ].mean(axis=1)
    return features


def _split_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(frame) < 100:
        raise ValueError("Servono almeno 100 esempi per la valutazione sperimentale")
    frame = frame.sort_values("date").reset_index(drop=True)
    train_end = int(len(frame) * 0.70)
    validation_end = int(len(frame) * 0.85)
    validation_start = frame.iloc[train_end]["date"]
    test_start = frame.iloc[validation_end]["date"]
    horizon = pd.Timedelta(days=2)
    train = frame.iloc[:train_end]
    validation = frame.iloc[train_end:validation_end]
    test = frame.iloc[validation_end:]
    # La target D+2 deve essere già nota alla data D della previsione successiva.
    train = train.loc[train["date"] + horizon <= validation_start]
    validation = validation.loc[validation["date"] + horizon <= test_start]
    if min(len(train), len(validation), len(test)) < 3:
        raise ValueError("Dati insufficienti dopo la separazione temporale D+2")
    return train, validation, test


def _validation_periods(
    validation: pd.DataFrame, actual: np.ndarray, predicted: np.ndarray,
) -> list[dict[str, Any]]:
    periods = []
    for indices in np.array_split(np.arange(len(validation)), 3):
        periods.append({
            "first_feature_date": validation.iloc[indices[0]]["date"].isoformat(),
            "last_feature_date": validation.iloc[indices[-1]]["date"].isoformat(),
            "samples": len(indices),
            "mae": float(mean_absolute_error(actual[indices], predicted[indices])),
        })
    return periods


def _rain_metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    actual_rain = actual > RAIN_THRESHOLD_MM
    predicted_rain = probability >= RAIN_PROBABILITY_THRESHOLD
    matrix = confusion_matrix(actual_rain, predicted_rain, labels=[False, True])
    return {
        "event": "precipitazione giornaliera > 0 mm",
        "probability_threshold": RAIN_PROBABILITY_THRESHOLD,
        "brier_score": float(brier_score_loss(actual_rain, probability)),
        "confusion_matrix": matrix.tolist(),
        "matrix_order": ["asciutto", "piovoso"],
    }


def run_experiments(hourly: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    """Seleziona candidati su validation e valuta sul test cronologico separato."""
    daily = aggregate_daily(hourly)
    if daily["date"].diff().dropna().ne(pd.Timedelta(days=1)).any():
        raise ValueError(
            "Esperimento impossibile: giornate complete mancanti nello storico; "
            "i lag e i target D+2 non sarebbero allineati"
        )
    frame, feature_columns = build_training_frame(hourly)
    train, validation, test = _split_frame(frame)
    x_train = _add_recent_features(train, feature_columns)
    x_validation = _add_recent_features(validation, feature_columns)
    x_test = _add_recent_features(test, feature_columns)
    output = settings.artifacts_dir / "experiments"
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "selection": "MAE sulla validation completa; il test non partecipa alla selezione",
        "validation_periods_note": (
            "Tre intervalli contigui di validation valutati con lo stesso modello "
            "addestrato sul training iniziale; non sono finestre rolling riaddestrate."
        ),
        "sample_counts": {
            "train": len(train), "validation": len(validation), "test": len(test)
        },
        "split_dates": {
            name: {"first_feature_date": subset["date"].iloc[0].isoformat(),
                   "last_feature_date": subset["date"].iloc[-1].isoformat()}
            for name, subset in (("train", train), ("validation", validation), ("test", test))
        },
        "targets": {},
        "limitations": [
            "I risultati dipendono da un solo split cronologico; i tre intervalli non sono backtest rolling.",
            "Lo storico è rianalisi Open-Meteo, non osservazioni indipendenti di una stazione.",
            "Le previsioni NWP storiche non sono utilizzate; il blend operativo non viene calibrato.",
            "I modelli sperimentali non sostituiscono il modello operativo e non sono esportati in ONNX.",
        ],
    }
    selected_models: dict[str, Any] = {}
    for target in TARGET_COLUMNS:
        y_train = train[f"target_{target}"]
        y_validation = validation[f"target_{target}"].to_numpy(dtype=float)
        y_test = test[f"target_{target}"].to_numpy(dtype=float)
        baseline_validation = validation[target].to_numpy(dtype=float)
        baseline_test = test[target].to_numpy(dtype=float)
        candidates: dict[str, Any] = {
            "random_forest_leaf_2": _forest(2),
            "random_forest_leaf_6": _forest(6),
            "extra_trees_leaf_4": ExtraTreesRegressor(
                n_estimators=100, min_samples_leaf=4, max_features=0.8,
                random_state=42, n_jobs=-1,
            ),
        }
        if target == "temperature_2m_max":
            candidates["residual_persistence"] = ResidualTemperatureModel(_forest(4))
        if target == "precipitation_sum":
            rainy = y_train > RAIN_THRESHOLD_MM
            if rainy.nunique() == 2:
                candidates["two_stage_rain"] = TwoStageRainModel(
                    RandomForestClassifier(
                        n_estimators=100, min_samples_leaf=4, max_features=0.8,
                        random_state=42, n_jobs=-1,
                    ),
                    _forest(4),
                )
            else:
                LOGGER.warning("Esperimento pioggia a due fasi non disponibile: manca una classe nel training")

        validation_results: dict[str, Any] = {}
        rain_model: TwoStageRainModel | None = None
        for name, model in candidates.items():
            model.fit(x_train, y_train)
            prediction = model.predict(x_validation)
            validation_results[name] = {
                **_regression_metrics(y_validation, prediction),
                "periods": _validation_periods(validation, y_validation, prediction),
            }
            if isinstance(model, TwoStageRainModel):
                rain_model = model

        selected_name = min(validation_results, key=lambda name: validation_results[name]["mae"])
        selected = candidates[selected_name]
        selected_models[target] = selected
        test_prediction = selected.predict(x_test)
        result: dict[str, Any] = {
            "unit": UNITS[target],
            "selected": selected_name,
            "validation_candidates": validation_results,
            "validation_persistence": _regression_metrics(y_validation, baseline_validation),
            "test_selected": _regression_metrics(y_test, test_prediction),
            "test_persistence": _regression_metrics(y_test, baseline_test),
        }
        if target == "precipitation_sum":
            if rain_model is None:
                result["rain_classification"] = {
                    "available": False,
                    "reason": "Il training non contiene entrambe le classi di precipitazione.",
                }
            else:
                rainy_test = y_test > RAIN_THRESHOLD_MM
                conditional_prediction = rain_model.amount_model.predict(x_test)
                result["rain_classification"] = {
                    "available": True,
                    "selected_for_regression": selected_name == "two_stage_rain",
                    **_rain_metrics(y_test, rain_model.predict_probability(x_test)),
                    "conditional_amount_mae_rainy_days_mm": (
                        float(mean_absolute_error(y_test[rainy_test], conditional_prediction[rainy_test]))
                        if rainy_test.any() else None
                    ),
                    "rainy_test_samples": int(rainy_test.sum()),
                }
        report["targets"][target] = result
        _save_test_plot(
            plots / f"{target}_test.png", target, UNITS[target], test["date"],
            y_test, test_prediction, baseline_test,
        )

    joblib.dump(
        {"models": selected_models, "feature_columns": list(x_train.columns)},
        output / "experimental_models.joblib",
    )
    (output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Risultati sperimentali salvati in %s", output)
    return report


def format_experiment_results(report: dict[str, Any]) -> str:
    lines = [
        "Esperimenti: candidato scelto sulla validation, metriche sul test",
        f"{'Target':<20} {'Modello':<26} {'MAE ML':>9} {'MAE baseline':>12} {'RMSE ML':>9} {'R2 ML':>8}",
    ]
    for target, result in report["targets"].items():
        score = result["test_selected"]
        lines.append(
            f"{target:<20} {result['selected']:<26} {score['mae']:>9.3f} "
            f"{result['test_persistence']['mae']:>12.3f} {score['rmse']:>9.3f} {score['r2']:>8.3f}"
        )
    rain = report["targets"]["precipitation_sum"]["rain_classification"]
    if rain["available"]:
        matrix = rain["confusion_matrix"]
        lines.extend([
            "Pioggia a due fasi: matrice test (soglia probabilita 0.5; righe reali, colonne previste)",
            "              asciutto  piovoso",
            f"asciutto      {matrix[0][0]:>8}  {matrix[0][1]:>7}",
            f"piovoso       {matrix[1][0]:>8}  {matrix[1][1]:>7}",
            f"Brier score: {rain['brier_score']:.3f}; "
            f"modello a due fasi selezionato: {'si' if rain['selected_for_regression'] else 'no'}",
        ])
    else:
        lines.append(f"Pioggia a due fasi non disponibile: {rain['reason']}")
    return "\n".join(lines)
