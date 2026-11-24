from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import rasterio
import numpy as np
from rasterio.windows import Window

from copernicus_flood.ingestion.registry import IngestionError


def write_spatial_patch_manifest(
    image_path: Path,
    label_path: Path,
    output_path: Path,
    *,
    patch_size: int = 256,
    min_valid_fraction: float = 0.8,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> dict[str, Any]:
    if patch_size <= 0 or not 0 < min_valid_fraction <= 1:
        raise IngestionError("patch_size e min_valid_fraction non sono validi.")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise IngestionError("Le proporzioni degli split devono essere tra 0 e 1.")
    if train_fraction + validation_fraction >= 1:
        raise IngestionError("La proporzione test risultante deve essere positiva.")

    records: list[dict[str, Any]] = []
    with rasterio.open(image_path) as image, rasterio.open(label_path) as labels:
        if (
            image.width != labels.width
            or image.height != labels.height
            or image.transform != labels.transform
            or image.crs != labels.crs
        ):
            raise IngestionError("Immagine e maschera non condividono la stessa griglia.")

        train_end = int(image.height * train_fraction)
        validation_end = int(
            image.height * (train_fraction + validation_fraction)
        )
        guard = patch_size

        for row_off in range(0, image.height - patch_size + 1, patch_size):
            if row_off + patch_size <= train_end - guard:
                split = "train"
            elif row_off >= train_end + guard and row_off + patch_size <= validation_end - guard:
                split = "validation"
            elif row_off >= validation_end + guard:
                split = "test"
            else:
                continue

            for col_off in range(0, image.width - patch_size + 1, patch_size):
                window = Window(col_off, row_off, patch_size, patch_size)
                target = labels.read(1, window=window)
                valid_pixels = int(np.count_nonzero(target != 255))
                total_pixels = patch_size * patch_size
                if valid_pixels / total_pixels < min_valid_fraction:
                    continue
                records.append(
                    {
                        "patch_id": f"r{row_off:07d}_c{col_off:07d}",
                        "split": split,
                        "row_off": row_off,
                        "col_off": col_off,
                        "height": patch_size,
                        "width": patch_size,
                        "valid_pixels": valid_pixels,
                        "positive_pixels": int(np.count_nonzero(target == 1)),
                        "negative_pixels": int(np.count_nonzero(target == 0)),
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                    }
                )

    if not records:
        raise IngestionError(
            "Nessuna patch soddisfa dimensione e frazione minima di pixel validi."
        )

    split_counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "validation", "test")
    }
    class_pixel_counts = {
        split: {
            "positive": sum(
                record["positive_pixels"]
                for record in records
                if record["split"] == split
            ),
            "negative": sum(
                record["negative_pixels"]
                for record in records
                if record["split"] == split
            ),
        }
        for split in ("train", "validation", "test")
    }
    empty_splits = [
        split for split, count in split_counts.items() if count == 0
    ]
    splits_without_positive = [
        split
        for split, counts in class_pixel_counts.items()
        if split_counts[split] > 0 and counts["positive"] == 0
    ]
    splits_without_negative = [
        split
        for split, counts in class_pixel_counts.items()
        if split_counts[split] > 0 and counts["negative"] == 0
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as manifest:
        for record in records:
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "manifest": str(output_path),
        "patch_count": len(records),
        "split_counts": split_counts,
        "class_pixel_counts": class_pixel_counts,
        "empty_splits": empty_splits,
        "splits_without_positive": splits_without_positive,
        "splits_without_negative": splits_without_negative,
        "spatial_split_note": (
            "Gli split sono blocchi spaziali dello stesso evento EMSR865; "
            "non misurano generalizzazione a eventi distinti."
        ),
    }
