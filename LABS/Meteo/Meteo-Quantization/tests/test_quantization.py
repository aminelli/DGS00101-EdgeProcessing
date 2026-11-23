from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from quant_weather.data import TARGET_COLUMNS, Dataset, load_dataset
from quant_weather.pipeline import train_export_quantize


def test_load_dataset_rejects_missing_targets(tmp_path):
    source = tmp_path / "features.csv"
    pd.DataFrame({"date": ["2025-01-01"], "feature": [1.0]}).to_csv(source, index=False)
    with pytest.raises(ValueError, match="Target mancanti"):
        load_dataset(source)


def test_train_exports_and_quantizes_mlp(tmp_path):
    sample_count = 120
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    dates = pd.Series(
        [start + timedelta(days=index) for index in range(sample_count)]
    )
    first = np.linspace(-1, 1, sample_count, dtype=np.float32)
    second = np.sin(np.linspace(0, 5, sample_count)).astype(np.float32)
    features = pd.DataFrame({"feature_a": first, "feature_b": second})
    targets = pd.DataFrame(
        {
            target: first * (index + 1) + second * 0.2
            for index, target in enumerate(TARGET_COLUMNS)
        }
    )
    persistence = pd.DataFrame(
        {
            target.removeprefix("target_"): targets[target].shift(1).fillna(0)
            for target in TARGET_COLUMNS
        }
    )
    dataset = Dataset(features, targets, persistence, dates)

    report = train_export_quantize(
        dataset,
        tmp_path / "artifacts",
        hidden_layers=(8,),
        max_iter=500,
    )

    assert set(report["targets"]) == set(TARGET_COLUMNS)
    for target, result in report["targets"].items():
        assert result["fp32_model"]["onnx_verified"] is True
        assert result["fp32_model"]["size_bytes"] > 0
        assert result["int8_model"]["size_bytes"] > 0
        assert (tmp_path / "artifacts" / result["int8_model"]["path"]).is_file()
        assert "mlp_int8" in result["metrics"]["test"]
        assert result["int8_model"]["max_abs_error_vs_fp32"] >= 0
    assert (tmp_path / "artifacts" / "quantization_report.json").is_file()
