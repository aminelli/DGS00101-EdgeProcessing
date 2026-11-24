from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.errors import RasterioError
import torch
from torch.utils.data import DataLoader

from copernicus_flood.dataset.manifest import load_training_manifest
from copernicus_flood.dataset.torch_dataset import (
    SatellitePatchDataset,
    close_dataset_handles,
)
from copernicus_flood.evaluation.metrics import (
    binary_confusion_matrix,
    metrics_from_confusion,
    render_confusion_matrix_svg,
)
from copernicus_flood.ingestion.registry import IngestionError
from copernicus_flood.models.flood.unet import UNet


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    dataset_info, records, manifest_sha256 = load_training_manifest(
        args.dataset_dir, args.patch_manifest
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("dataset_manifest_sha256") != manifest_sha256:
        raise IngestionError(
            "Il checkpoint non corrisponde al manifesto del dataset selezionato."
        )
    if checkpoint.get("activation_code") != "EMSR865":
        raise IngestionError("Il checkpoint non è stato addestrato su EMSR865.")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict) or model_config.get("architecture") != "unet":
        raise IngestionError("Architettura checkpoint non supportata.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise IngestionError("CUDA è stato richiesto ma non è disponibile.")
    model = UNet(
        in_channels=int(model_config["in_channels"]),
        num_classes=int(model_config["num_classes"]),
        base_channels=int(model_config["base_channels"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    test_records = [record for record in records if record["split"] == "test"]
    dataset = SatellitePatchDataset(
        test_records,
        channel_mean=checkpoint["channel_mean"],
        channel_std=checkpoint["channel_std"],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    total_confusion = np.zeros((2, 2), dtype=np.int64)
    patch_reports: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            targets = batch["target"].cpu().numpy()
            predictions = model(images).argmax(dim=1).cpu().numpy()
            for patch_id, prediction, target in zip(
                batch["patch_id"], predictions, targets
            ):
                confusion = binary_confusion_matrix(prediction, target)
                total_confusion += confusion
                patch_reports.append(
                    {
                        "patch_id": patch_id,
                        "metrics": metrics_from_confusion(confusion),
                    }
                )
    close_dataset_handles(dataset)

    metrics = metrics_from_confusion(total_confusion)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "activation_code": "EMSR865",
        "split": "test",
        "metrics": metrics,
        "patch_count": len(test_records),
        "per_patch": patch_reports,
        "model_checkpoint": str(args.checkpoint),
        "model_checkpoint_sha256": _sha256_file(args.checkpoint),
        "dataset_manifest_sha256": manifest_sha256,
        "negative_label_assumption": dataset_info["negative_label_assumption"],
        "evaluation_scope": dataset_info["evaluation_scope"],
        "generalization_warning": (
            "Il test è un holdout spaziale della sola attivazione EMSR865; "
            "non misura la generalizzazione a eventi distinti."
        ),
    }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    render_confusion_matrix_svg(
        total_confusion,
        str(output_dir / "confusion_matrix.svg"),
        title="EMSR865 - split test",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valuta U-Net Flood sullo split spaziale test di EMSR865."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--patch-manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch-size deve essere positivo e workers non negativo.")
    try:
        result = evaluate_model(args)
    except (
        IngestionError,
        OSError,
        ValueError,
        RuntimeError,
        RasterioError,
    ) as error:
        print(f"ERRORE: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
