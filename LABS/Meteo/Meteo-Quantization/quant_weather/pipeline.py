"""Training MLP, conversione ONNX, quantizzazione INT8 e verifica."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from quant_weather.data import TARGET_COLUMNS, Dataset

LOGGER = logging.getLogger(__name__)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def _benchmark(predict: Callable[[np.ndarray], Any], features: np.ndarray) -> float:
    """Misura il tempo medio per batch dopo un riscaldamento."""
    predict(features)
    repeats = 20
    started = perf_counter()
    for _ in range(repeats):
        predict(features)
    return (perf_counter() - started) * 1000 / repeats


def _fit_predictor(
    *,
    hidden_layers: tuple[int, ...],
    max_iter: int,
    random_state: int,
) -> Any:
    """Crea un modello di regressione con layer MatMul/Gemm convertibili in ONNX."""
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            alpha=0.01,
            learning_rate_init=0.001,
            max_iter=max_iter,
            tol=1e-4,
            random_state=random_state,
        ),
    )


def train_export_quantize(
    dataset: Dataset,
    output_dir: str | Path,
    *,
    hidden_layers: tuple[int, ...] = (64, 32),
    max_iter: int = 1200,
    random_state: int = 42,
    quality_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Allena modelli distinti, esporta ONNX/INT8 e confronta i risultati sul test."""
    if len(dataset.features) < 100:
        raise ValueError("Servono almeno 100 esempi per uno split temporale significativo")
    if not hidden_layers or any(width < 1 for width in hidden_layers):
        raise ValueError("Ogni layer nascosto deve avere almeno un neurone")
    if max_iter < 1 or quality_tolerance < 0:
        raise ValueError("max_iter e quality_tolerance non possono essere negativi")

    count = len(dataset.features)
    train_end = int(count * 0.70)
    validation_end = int(count * 0.85)
    if train_end == 0 or validation_end <= train_end or validation_end >= count:
        raise ValueError("Dataset insufficiente per lo split cronologico 70/15/15")

    x_train = dataset.features.iloc[:train_end].to_numpy(dtype=np.float32)
    x_validation = dataset.features.iloc[train_end:validation_end].to_numpy(dtype=np.float32)
    x_test = dataset.features.iloc[validation_end:].to_numpy(dtype=np.float32)
    output_path = Path(output_dir).expanduser().resolve()
    float_dir = output_path / "onnx"
    quantized_dir = output_path / "quantized"
    float_dir.mkdir(parents=True, exist_ok=True)
    quantized_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "model_type": "StandardScaler + scikit-learn MLPRegressor",
        "onnx_quantization_method": "ONNX Runtime dynamic quantization, QInt8",
        "feature_columns": list(dataset.features.columns),
        "hidden_layers": list(hidden_layers),
        "max_iter": max_iter,
        "random_state": random_state,
        "quality_tolerance_relative_rmse": quality_tolerance,
        "split": {
            "train": train_end,
            "validation": validation_end - train_end,
            "test": count - validation_end,
            "method": "chronological 70/15/15",
        },
        "targets": {},
        "limitations": [
            "Il modello MLP è un esperimento separato, non sostituisce la Random Forest del progetto esistente.",
            "La quantizzazione e la compatibilità ONNX non implicano che il modello migliori il baseline meteorologico.",
            "I tempi misurati sono batch CPU in questo ambiente e non rappresentano le prestazioni di un dispositivo edge.",
        ],
    }
    models: dict[str, Any] = {}

    for target_index, target in enumerate(TARGET_COLUMNS):
        y_all = dataset.targets[target].to_numpy(dtype=np.float32)
        y_train = y_all[:train_end]
        y_validation = y_all[train_end:validation_end]
        y_test = y_all[validation_end:]
        model = _fit_predictor(
            hidden_layers=hidden_layers,
            max_iter=max_iter,
            random_state=random_state,
        )
        model.fit(x_train, y_train)
        models[target] = model

        validation_prediction = model.predict(x_validation).astype(np.float32)
        float_prediction = model.predict(x_test).astype(np.float32)
        baseline_prediction = dataset.persistence[target.removeprefix("target_")].to_numpy(
            dtype=np.float32
        )[validation_end:]

        model_name = target.removeprefix("target_")
        float_path = float_dir / f"{model_name}.onnx"
        quantized_path = quantized_dir / f"{model_name}.int8.onnx"
        converted = convert_sklearn(
            model,
            initial_types=[
                ("features", FloatTensorType([None, dataset.features.shape[1]]))
            ],
            target_opset=17,
        )
        float_path.write_bytes(converted.SerializeToString())
        quantize_dynamic(
            model_input=str(float_path),
            model_output=str(quantized_path),
            weight_type=QuantType.QInt8,
        )

        import onnxruntime as ort

        float_session = ort.InferenceSession(
            str(float_path), providers=["CPUExecutionProvider"]
        )
        quantized_session = ort.InferenceSession(
            str(quantized_path), providers=["CPUExecutionProvider"]
        )
        float_onnx_predict = lambda values: float_session.run(
            None, {"features": values}
        )[0].reshape(-1)
        quantized_onnx_predict = lambda values: quantized_session.run(
            None, {"features": values}
        )[0].reshape(-1)

        onnx_float_prediction = float_onnx_predict(x_test)
        quantized_prediction = quantized_onnx_predict(x_test)
        onnx_max_abs_error = float(np.max(np.abs(onnx_float_prediction - float_prediction)))
        if not np.allclose(onnx_float_prediction, float_prediction, atol=1e-4, rtol=1e-4):
            raise RuntimeError(f"Verifica ONNX FP32 non superata per {target}")

        float_rmse = float(np.sqrt(mean_squared_error(y_test, float_prediction)))
        quantized_rmse = float(np.sqrt(mean_squared_error(y_test, quantized_prediction)))
        relative_rmse_change = (quantized_rmse - float_rmse) / max(float_rmse, 1e-8)
        metrics = {
            "validation": {
                "mlp_float": _metrics(y_validation, validation_prediction),
            },
            "test": {
                "baseline_persistence": _metrics(y_test, baseline_prediction),
                "mlp_float": _metrics(y_test, float_prediction),
                "mlp_int8": _metrics(y_test, quantized_prediction),
            },
        }
        report["targets"][target] = {
            "training": {
                "iterations": int(model[-1].n_iter_),
                "max_iter": max_iter,
                "converged_before_max_iter": bool(model[-1].n_iter_ < max_iter),
            },
            "fp32_model": {
                "path": str(float_path.relative_to(output_path)),
                "size_bytes": float_path.stat().st_size,
                "test_batch_inference_ms": _benchmark(
                    lambda values: model.predict(values), x_test
                ),
                "onnx_batch_inference_ms": _benchmark(float_onnx_predict, x_test),
                "max_abs_error_vs_python": onnx_max_abs_error,
                "onnx_verified": True,
            },
            "int8_model": {
                "path": str(quantized_path.relative_to(output_path)),
                "size_bytes": quantized_path.stat().st_size,
                "test_batch_inference_ms": _benchmark(quantized_onnx_predict, x_test),
                "max_abs_error_vs_fp32": float(
                    np.max(np.abs(quantized_prediction - onnx_float_prediction))
                ),
                "rmse_relative_change_vs_fp32": relative_rmse_change,
                "quantization_fidelity_accepted": (
                    relative_rmse_change <= quality_tolerance
                ),
            },
            "metrics": metrics,
        }

    joblib.dump(
        {
            "models": models,
            "feature_columns": list(dataset.features.columns),
            "target_columns": TARGET_COLUMNS,
            "hidden_layers": hidden_layers,
        },
        output_path / "python_models.joblib",
    )
    (output_path / "quantization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Modelli ONNX e INT8 e report salvati in %s", output_path)
    return report
