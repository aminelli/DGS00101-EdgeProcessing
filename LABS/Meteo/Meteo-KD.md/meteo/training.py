"""Training, confronto cronologico e valutazione dei modelli."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from meteo.config import Settings
from meteo.features import TARGET_COLUMNS, build_training_frame

LOGGER = logging.getLogger(__name__)


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def _benchmark_prediction(
    predict: Callable[[Any], Any], values: Any, repeats: int = 20
) -> float:
    predict(values)
    started = perf_counter()
    for _ in range(repeats):
        predict(values)
    return (perf_counter() - started) * 1000 / repeats


def _save_test_plot(
    path: Path,
    target: str,
    unit: str,
    dates: pd.Series,
    actual: np.ndarray,
    predicted: np.ndarray,
    baseline: np.ndarray,
) -> None:
    """Salva il confronto cronologico e osservato-previsto per il test."""
    figure = Figure(figsize=(13, 5), layout="constrained")
    FigureCanvasAgg(figure)
    series_ax, scatter_ax = figure.subplots(1, 2)
    target_dates = dates + pd.Timedelta(days=2)
    series_ax.plot(target_dates, actual, label="Storico", linewidth=1.3)
    series_ax.plot(target_dates, predicted, label="ML", linewidth=1.1)
    series_ax.plot(target_dates, baseline, label="Persistenza", linewidth=1, alpha=0.7)
    series_ax.set_title("Serie temporale - test")
    series_ax.set_xlabel("Data target (UTC)")
    series_ax.set_ylabel(f"{target} ({unit})")
    series_ax.tick_params(axis="x", labelrotation=30)
    series_ax.legend()
    series_ax.grid(alpha=0.25)

    scatter_ax.scatter(actual, predicted, s=14, alpha=0.55, label="ML")
    lower = min(float(actual.min()), float(predicted.min()))
    upper = max(float(actual.max()), float(predicted.max()))
    scatter_ax.plot([lower, upper], [lower, upper], "k--", label="Previsione ideale")
    scatter_ax.set_title("Storico vs previsione ML - test")
    scatter_ax.set_xlabel(f"Storico ({unit})")
    scatter_ax.set_ylabel(f"Previsione ML ({unit})")
    scatter_ax.legend()
    scatter_ax.grid(alpha=0.25)
    figure.suptitle(f"{target} - orizzonte D+2")
    figure.savefig(path, dpi=120)
    figure.clear()


def format_test_results(metrics: dict[str, Any]) -> str:
    """Formatta le metriche test ML e baseline in una tabella leggibile."""
    rows = []
    for target, result in metrics["targets"].items():
        for label, key in (
            ("ML", "machine_learning"),
            ("Persistenza", "persistence_baseline"),
        ):
            scores = result[key]
            rows.append(
                (
                    target,
                    result["unit"],
                    label,
                    *(f"{scores[name]:.3f}" for name in ("mae", "rmse", "r2")),
                )
            )
    headers = ("Target", "Unita", "Modello", "MAE", "RMSE", "R2")
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]

    def format_row(row: tuple[str, ...]) -> str:
        return " | ".join(
            value.ljust(widths[i]) if i < 3 else value.rjust(widths[i])
            for i, value in enumerate(row)
        )

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join(
        ["Risultati sul test (orizzonte D+2)", format_row(headers), separator]
        + [format_row(row) for row in rows]
        + ["Matrice di confusione non applicabile: i target sono continui."]
    )


def train_models(hourly: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    frame, feature_columns = build_training_frame(hourly)
    if len(frame) < 100:
        raise ValueError(
            f"Solo {len(frame)} esempi validi: ne servono almeno 100. "
            "Verificare completezza e durata dei dati orari."
        )

    count = len(frame)
    train_end = int(count * 0.70)
    validation_end = int(count * 0.85)
    train = frame.iloc[:train_end]
    validation = frame.iloc[train_end:validation_end]
    test = frame.iloc[validation_end:]
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Suddivisione cronologica impossibile: dataset troppo piccolo")

    models: dict[str, RandomForestRegressor] = {}
    metrics: dict[str, Any] = {
        "description": (
            "Split cronologico 70/15/15. La validazione serve per controllo; "
            "i risultati finali sono calcolati sul test."
        ),
        "sample_counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "targets": {},
        "limitations": [
            "Il dataset storico Open-Meteo è una serie di rianalisi/modello, non una raccolta di misure di una singola stazione.",
            "La valutazione storica del modello ML è distinta dalla previsione NWP operativa.",
            "Non è disponibile qui un archivio di previsioni NWP storiche corrispondenti: il blending ibrido in inferenza non è calibrato né presentato come validato.",
        ],
    }

    plots_dir = settings.artifacts_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for target in TARGET_COLUMNS:
        target_name = f"target_{target}"
        x_train = train[feature_columns]
        y_train = train[target_name]
        x_validation = validation[feature_columns]
        y_validation = validation[target_name].to_numpy(dtype=float)
        x_test = test[feature_columns]
        y_test = test[target_name].to_numpy(dtype=float)

        model = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        models[target] = model

        validation_prediction = model.predict(x_validation)
        ml_prediction = model.predict(x_test)
        # La baseline persiste il valore del giorno corrente fino alla data target D+2.
        current_target = target.removeprefix("target_")
        validation_persistence = validation[current_target].to_numpy(dtype=float)
        persistence_prediction = test[current_target].to_numpy(dtype=float)
        metrics["targets"][target] = {
            "unit": {"temperature_2m_max": "°C", "precipitation_sum": "mm", "wind_speed_10m_max": "km/h"}[target],
            "validation": {
                "machine_learning": _regression_metrics(y_validation, validation_prediction),
                "persistence_baseline": _regression_metrics(
                    y_validation, validation_persistence
                ),
            },
            "machine_learning": _regression_metrics(y_test, ml_prediction),
            "persistence_baseline": _regression_metrics(y_test, persistence_prediction),
        }
        plot_path = plots_dir / f"{target}_test.png"
        _save_test_plot(
            plot_path,
            target,
            metrics["targets"][target]["unit"],
            test["date"],
            y_test,
            ml_prediction,
            persistence_prediction,
        )
        LOGGER.info("Grafico test salvato in %s", plot_path)

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "models": models,
        "feature_columns": feature_columns,
        "target_columns": TARGET_COLUMNS,
        "location": {
            "name": settings.location_name,
            "latitude": settings.latitude,
            "longitude": settings.longitude,
        },
        "forecast_horizon": "Tomorrow in UTC; target is two days after the latest complete UTC day",
    }
    joblib.dump(artifact, settings.model_path)
    settings.metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Modelli salvati in %s", settings.model_path)
    LOGGER.info("Metriche salvate in %s", settings.metrics_path)
    return metrics


def export_onnx(settings: Settings, hourly: pd.DataFrame) -> dict[str, Any]:
    """Esporta modelli ONNX e prova la quantizzazione solo se gli extra sono installati."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "Esportazione ONNX non disponibile. Installare gli extra con "
            "`pip install -e \".[edge]\"`."
        ) from exc

    artifact = joblib.load(settings.model_path)
    frame, _ = build_training_frame(hourly)
    if frame.empty:
        raise ValueError("Nessun dato disponibile per verificare l'esportazione ONNX")
    x_check_frame = frame[artifact["feature_columns"]].tail(256)
    x_check = x_check_frame.to_numpy(dtype=np.float32)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "benchmark_batch_size": len(x_check),
        "benchmark_repeats": 20,
        "onnx": {},
        "quantization": {},
    }
    for target, model in artifact["models"].items():
        target_values = frame[f"target_{target}"].tail(len(x_check)).to_numpy(dtype=np.float32)
        model_path = settings.artifacts_dir / f"{target}.onnx"
        model_onnx = convert_sklearn(
            model,
            initial_types=[("features", FloatTensorType([None, len(artifact["feature_columns"])]))],
            target_opset=17,
        )
        model_path.write_bytes(model_onnx.SerializeToString())
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        onnx_prediction = session.run(None, {"features": x_check})[0].reshape(-1)
        sklearn_prediction = model.predict(x_check_frame)
        max_abs_error = float(np.max(np.abs(onnx_prediction - sklearn_prediction)))
        sklearn_inference_ms = _benchmark_prediction(model.predict, x_check_frame)
        onnx_inference_ms = _benchmark_prediction(
            lambda values: session.run(None, {"features": values})[0], x_check
        )
        report["onnx"][target] = {
            "path": str(model_path.relative_to(settings.artifacts_dir)),
            "size_bytes": model_path.stat().st_size,
            "max_absolute_error": max_abs_error,
            "verified": bool(np.allclose(onnx_prediction, sklearn_prediction, atol=1e-4)),
            "sklearn_batch_inference_ms": sklearn_inference_ms,
            "onnx_batch_inference_ms": onnx_inference_ms,
        }
        if not report["onnx"][target]["verified"]:
            raise RuntimeError(f"Verifica ONNX non superata per {target}: errore {max_abs_error}")

        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quantized_path = settings.artifacts_dir / f"{target}.int8.onnx"
            quantize_dynamic(str(model_path), str(quantized_path), weight_type=QuantType.QInt8)
            quantized_session = ort.InferenceSession(
                str(quantized_path), providers=["CPUExecutionProvider"]
            )
            quantized_prediction = quantized_session.run(
                None, {"features": x_check}
            )[0].reshape(-1)
            original_rmse = float(
                np.sqrt(np.mean((sklearn_prediction - target_values) ** 2))
            )
            quantized_rmse = float(
                np.sqrt(np.mean((quantized_prediction - target_values) ** 2))
            )
            relative_degradation = (
                quantized_rmse - original_rmse
            ) / max(original_rmse, 1e-8)
            accepted = relative_degradation <= 0.05
            report["quantization"][target] = {
                "path": str(quantized_path.relative_to(settings.artifacts_dir)) if accepted else None,
                "size_bytes": quantized_path.stat().st_size,
                "original_rmse": original_rmse,
                "quantized_rmse": quantized_rmse,
                "relative_rmse_degradation": relative_degradation,
                "inference_ms_per_batch": _benchmark_prediction(
                    lambda values: quantized_session.run(None, {"features": values})[0],
                    x_check,
                ),
                "accepted": accepted,
                "reason": None if accepted else "Peggioramento RMSE superiore al 5%; artefatto non promosso.",
            }
            if not accepted:
                quantized_path.unlink(missing_ok=True)
        except (ValueError, RuntimeError, OSError) as exc:
            report["quantization"][target] = {
                "path": None,
                "accepted": False,
                "reason": f"Quantizzazione non supportata o non verificabile: {exc}",
            }

    report_path = settings.artifacts_dir / "export_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
