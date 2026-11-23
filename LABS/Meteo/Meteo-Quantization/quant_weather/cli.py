"""Interfaccia da riga di comando per il progetto di quantizzazione."""

from __future__ import annotations

import argparse
import logging

from quant_weather.data import load_dataset
from quant_weather.pipeline import train_export_quantize


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Allena un MLP meteo compatibile con ONNX e ne crea la versione INT8"
    )
    parser.add_argument(
        "--data-csv",
        required=True,
        help="CSV training_features.csv; il file sorgente viene letto senza modificarlo",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Cartella degli artefatti, relativa alla directory corrente se non assoluta",
    )
    parser.add_argument(
        "--hidden-layers",
        nargs="+",
        type=int,
        default=[64, 32],
        help="Numero di neuroni per layer MLP (default: 64 32)",
    )
    parser.add_argument("--max-iter", type=int, default=1200)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--quality-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    dataset = load_dataset(args.data_csv)
    report = train_export_quantize(
        dataset,
        args.output_dir,
        hidden_layers=tuple(args.hidden_layers),
        max_iter=args.max_iter,
        random_state=args.random_state,
        quality_tolerance=args.quality_tolerance,
    )
    print(f"Report: {args.output_dir}/quantization_report.json")
    for target, result in report["targets"].items():
        fp32 = result["metrics"]["test"]["mlp_float"]
        int8 = result["metrics"]["test"]["mlp_int8"]
        baseline = result["metrics"]["test"]["baseline_persistence"]
        print(
            f"{target}: MAE FP32={fp32['mae']:.3f}, INT8={int8['mae']:.3f}, "
            f"baseline={baseline['mae']:.3f}, "
            "quantization fidelity accepted="
            f"{result['int8_model']['quantization_fidelity_accepted']}"
        )


if __name__ == "__main__":
    main()
