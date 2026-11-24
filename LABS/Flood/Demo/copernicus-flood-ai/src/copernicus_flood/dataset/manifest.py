from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from copernicus_flood.ingestion.registry import IngestionError


def load_training_manifest(
    dataset_dir: Path,
    patch_manifest: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    dataset_path = dataset_dir / "dataset_manifest.json"
    try:
        dataset_info = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IngestionError(f"Manifesto dataset non leggibile: {dataset_path}") from error
    if not isinstance(dataset_info, dict):
        raise IngestionError("Il manifesto dataset deve essere un oggetto JSON.")
    if dataset_info.get("dataset_status") != "ready_for_training":
        raise IngestionError(
            "Training bloccato: dataset_status non è ready_for_training."
        )
    if dataset_info.get("activation_code") != "EMSR865":
        raise IngestionError("Il dataset non è limitato all'attivazione EMSR865.")
    if dataset_info.get("negative_label_assumption") != "aoi_background_is_negative":
        raise IngestionError("Assunzione sulle etichette negative assente.")
    association = dataset_info.get("scene_association")
    if not isinstance(association, dict) or association.get("status") != "verified":
        raise IngestionError("Associazione scena-prodotto non verificata.")

    manifest_path = patch_manifest or Path(
        str(dataset_info.get("patch_manifest", dataset_dir / "patches.jsonl"))
    )
    if not manifest_path.is_absolute() and not manifest_path.is_file():
        manifest_path = dataset_dir / manifest_path
    try:
        records = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise IngestionError(f"Manifesto patch non leggibile: {manifest_path}") from error
    if not records or any(not isinstance(record, dict) for record in records):
        raise IngestionError("Il manifesto patch è vuoto o contiene record non validi.")

    patch_summary = dataset_info.get("patches")
    if not isinstance(patch_summary, dict):
        raise IngestionError("Riepilogo patch assente nel manifesto dataset.")
    expected_count = patch_summary.get("patch_count")
    if expected_count != len(records):
        raise IngestionError(
            "Il numero di patch non corrisponde al riepilogo del manifesto dataset."
        )
    by_split = {
        split: [record for record in records if record.get("split") == split]
        for split in ("train", "validation", "test")
    }
    if any(record.get("split") not in by_split for record in records):
        raise IngestionError("Il manifesto patch contiene uno split non supportato.")
    for split, split_records in by_split.items():
        if not split_records:
            raise IngestionError(f"Split {split} vuoto: training/valutazione bloccati.")
        if not any(record.get("positive_pixels", 0) > 0 for record in split_records):
            raise IngestionError(f"Split {split} privo di pixel allagati.")
        if not any(record.get("negative_pixels", 0) > 0 for record in split_records):
            raise IngestionError(f"Split {split} privo di pixel background.")
    for record in records:
        for key in ("image_path", "label_path", "patch_id"):
            if key not in record:
                raise IngestionError(f"Campo {key} assente in un record patch.")

    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    return dataset_info, records, digest
