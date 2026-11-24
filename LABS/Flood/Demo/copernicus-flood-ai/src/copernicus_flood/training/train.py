from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.errors import RasterioError
import torch
from torch import nn
from torch.utils.data import DataLoader

from copernicus_flood.dataset.manifest import load_training_manifest
from copernicus_flood.dataset.torch_dataset import (
    SatellitePatchDataset,
    close_dataset_handles,
    compute_channel_statistics,
)
from copernicus_flood.evaluation.metrics import (
    binary_confusion_matrix,
    metrics_from_confusion,
)
from copernicus_flood.ingestion.registry import IngestionError
from copernicus_flood.models.flood.unet import UNet


def _select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise IngestionError("CUDA è stato richiesto ma non è disponibile.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_batches = 0
    confusion = np.zeros((2, 2), dtype=np.int64)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        targets = batch["target"].to(device, non_blocking=device.type == "cuda")
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if scaler is not None and scaler.is_enabled()
                else contextlib.nullcontext()
            )
            with autocast:
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += float(loss.detach().item())
        total_batches += 1
        predictions = logits.detach().argmax(dim=1).cpu().numpy()
        confusion += binary_confusion_matrix(
            predictions, targets.detach().cpu().numpy()
        )
    if total_batches == 0:
        raise IngestionError("Il DataLoader non ha prodotto batch.")
    return total_loss / total_batches, confusion


def train_model(args: argparse.Namespace) -> dict[str, Any]:
    dataset_info, records, manifest_sha256 = load_training_manifest(
        args.dataset_dir, args.patch_manifest
    )
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise IngestionError("Epoch, batch size e patience devono essere positivi.")
    if args.learning_rate <= 0 or args.workers < 0:
        raise IngestionError("Learning rate e numero di worker non sono validi.")

    _set_seed(args.seed)
    device = _select_device(args.device)
    train_records = [record for record in records if record["split"] == "train"]
    validation_records = [
        record for record in records if record["split"] == "validation"
    ]

    statistics_dataset = SatellitePatchDataset(train_records)
    try:
        channel_mean, channel_std = compute_channel_statistics(statistics_dataset)
    finally:
        close_dataset_handles(statistics_dataset)

    train_dataset = SatellitePatchDataset(
        train_records,
        augment=True,
        channel_mean=channel_mean,
        channel_std=channel_std,
    )
    validation_dataset = SatellitePatchDataset(
        validation_records,
        channel_mean=channel_mean,
        channel_std=channel_std,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=generator,
        persistent_workers=args.workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    model_config = {
        "architecture": "unet",
        "in_channels": 2,
        "num_classes": 2,
        "base_channels": args.base_channels,
    }
    model = UNet(**{key: value for key, value in model_config.items() if key != "architecture"})
    model.to(device)
    negative_pixels = sum(record["negative_pixels"] for record in train_records)
    positive_pixels = sum(record["positive_pixels"] for record in train_records)
    total_class_pixels = negative_pixels + positive_pixels
    class_weights = torch.tensor(
        [
            total_class_pixels / (2 * negative_pixels),
            total_class_pixels / (2 * positive_pixels),
        ],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    start_epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if checkpoint.get("dataset_manifest_sha256") != manifest_sha256:
            raise IngestionError(
                "Il checkpoint appartiene a un manifesto dataset diverso."
            )
        if checkpoint.get("model_config") != model_config:
            raise IngestionError("La configurazione del modello non coincide.")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(checkpoint["best_validation_loss"])
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "training_history.json"
    history: list[dict[str, Any]] = []
    if args.resume is not None and history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_confusion = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
        )
        with torch.no_grad():
            validation_loss, validation_confusion = _run_epoch(
                model, validation_loader, criterion, device
            )
        improved = validation_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_metrics": metrics_from_confusion(train_confusion),
            "validation_metrics": metrics_from_confusion(validation_confusion),
        }
        history.append(epoch_record)
        history_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        checkpoint_payload = {
            "checkpoint_version": 1,
            "epoch": epoch,
            "best_validation_loss": best_validation_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "model_config": model_config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "channel_mean": channel_mean,
            "channel_std": channel_std,
            "class_weights": class_weights.cpu().tolist(),
            "seed": args.seed,
            "dataset_manifest_sha256": manifest_sha256,
            "activation_code": "EMSR865",
            "evaluation_scope": dataset_info["evaluation_scope"],
            "negative_label_assumption": dataset_info[
                "negative_label_assumption"
            ],
        }
        _atomic_checkpoint(args.output_dir / "checkpoint_latest.pt", checkpoint_payload)
        if improved:
            _atomic_checkpoint(args.output_dir / "checkpoint_best.pt", checkpoint_payload)
        print(
            f"epoch={epoch + 1}/{args.epochs} train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f}"
        )
        if epochs_without_improvement >= args.patience:
            break

    close_dataset_handles(train_dataset)
    close_dataset_handles(validation_dataset)
    return {
        "checkpoint_best": str(args.output_dir / "checkpoint_best.pt"),
        "checkpoint_latest": str(args.output_dir / "checkpoint_latest.pt"),
        "history": str(history_path),
        "epochs_completed": len(history),
        "best_validation_loss": best_validation_loss,
        "device": str(device),
        "activation_code": "EMSR865",
        "evaluation_scope": dataset_info["evaluation_scope"],
        "negative_label_assumption": dataset_info["negative_label_assumption"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Addestra U-Net Flood sul solo dataset EMSR865 verificato."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--patch-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    try:
        result = train_model(args)
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
