from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

from copernicus_flood.ingestion.registry import IngestionError


class SatellitePatchDataset(Dataset[dict[str, Any]]):
    """Dataset lazy per patch; converte ampiezze grezze con log1p e usa ignore=255."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        augment: bool = False,
        channel_mean: list[float] | None = None,
        channel_std: list[float] | None = None,
    ) -> None:
        if not records:
            raise IngestionError("Lo split richiesto non contiene patch.")
        if (channel_mean is None) != (channel_std is None):
            raise IngestionError("Media e deviazione standard vanno fornite insieme.")
        if channel_mean is not None and (
            len(channel_mean) != 2
            or channel_std is None
            or len(channel_std) != 2
        ):
            raise IngestionError("La normalizzazione deve avere due canali VV/VH.")
        self.records = records
        self.augment = augment
        self.channel_mean = (
            np.asarray(channel_mean, dtype=np.float32) if channel_mean else None
        )
        self.channel_std = (
            np.asarray(channel_std, dtype=np.float32) if channel_std else None
        )
        self._handles: dict[str, rasterio.DatasetReader] = {}
        self._process_id = os.getpid()

    def __len__(self) -> int:
        return len(self.records)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_process_id"] = None
        return state

    def _dataset(self, path: str) -> rasterio.DatasetReader:
        process_id = os.getpid()
        if self._process_id != process_id:
            for dataset in self._handles.values():
                dataset.close()
            self._handles.clear()
            self._process_id = process_id
        dataset = self._handles.get(path)
        if dataset is None or dataset.closed:
            dataset = rasterio.open(path)
            self._handles[path] = dataset
        return dataset

    def read_raw(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        record = self.records[index]
        image_path = Path(record["image_path"])
        label_path = Path(record["label_path"])
        window = Window(
            int(record["col_off"]),
            int(record["row_off"]),
            int(record["width"]),
            int(record["height"]),
        )
        image_dataset = self._dataset(str(image_path))
        label_dataset = self._dataset(str(label_path))
        if image_dataset.count != 2 or label_dataset.count != 1:
            raise IngestionError("Stack atteso a due bande e maschera a una banda.")
        raw = image_dataset.read((1, 2), window=window).astype(np.float32)
        target = label_dataset.read(1, window=window).astype(np.uint8)
        input_valid = (
            (image_dataset.read_masks(1, window=window) > 0)
            & (image_dataset.read_masks(2, window=window) > 0)
            & np.isfinite(raw).all(axis=0)
        )
        for channel in range(2):
            if image_dataset.nodata is not None:
                input_valid &= raw[channel] != image_dataset.nodata
            raw[channel] = np.log1p(np.maximum(raw[channel], 0))
            raw[channel][~input_valid] = 0
        target[~input_valid] = 255
        if not np.isin(target, (0, 1, 255)).all():
            raise IngestionError("La maschera contiene classi diverse da 0, 1 e 255.")
        return raw, target

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw, target = self.read_raw(index)
        if self.channel_mean is None or self.channel_std is None:
            raise IngestionError("Statistiche di normalizzazione non inizializzate.")
        image = (raw - self.channel_mean[:, None, None]) / self.channel_std[:, None, None]
        image_tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))
        target_tensor = torch.from_numpy(np.ascontiguousarray(target, dtype=np.int64))

        if self.augment:
            if torch.rand(()) < 0.5:
                image_tensor = torch.flip(image_tensor, dims=(-1,))
                target_tensor = torch.flip(target_tensor, dims=(-1,))
            if torch.rand(()) < 0.5:
                image_tensor = torch.flip(image_tensor, dims=(-2,))
                target_tensor = torch.flip(target_tensor, dims=(-2,))
            rotations = int(torch.randint(0, 4, ()).item())
            if rotations:
                image_tensor = torch.rot90(image_tensor, rotations, dims=(-2, -1))
                target_tensor = torch.rot90(target_tensor, rotations, dims=(-2, -1))

        return {
            "image": image_tensor,
            "target": target_tensor,
            "patch_id": str(self.records[index]["patch_id"]),
        }


def compute_channel_statistics(
    dataset: SatellitePatchDataset,
) -> tuple[list[float], list[float]]:
    sums = np.zeros(2, dtype=np.float64)
    squared_sums = np.zeros(2, dtype=np.float64)
    pixel_count = 0
    for index in range(len(dataset)):
        image, target = dataset.read_raw(index)
        valid = target != 255
        if not np.any(valid):
            continue
        values = image[:, valid].astype(np.float64)
        sums += values.sum(axis=1)
        squared_sums += np.square(values).sum(axis=1)
        pixel_count += int(values.shape[1])
    if pixel_count == 0:
        raise IngestionError("Nessun pixel valido nello split di training.")
    means = sums / pixel_count
    variances = np.maximum(squared_sums / pixel_count - np.square(means), 0)
    standard_deviations = np.sqrt(variances)
    if np.any(standard_deviations < 1e-6):
        raise IngestionError("Deviazione standard nulla nello split di training.")
    return means.tolist(), standard_deviations.tolist()


def close_dataset_handles(dataset: SatellitePatchDataset) -> None:
    for handle in dataset._handles.values():
        handle.close()
    dataset._handles.clear()
