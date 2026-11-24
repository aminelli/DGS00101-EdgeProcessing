from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from copernicus_flood.dataset.patches import write_spatial_patch_manifest


def test_patch_manifest_reports_empty_spatial_splits(tmp_path: Path):
    image_path = tmp_path / "image.tif"
    label_path = tmp_path / "labels.tif"
    transform = from_origin(0, 64, 1, 1)
    for path, count, dtype, data, nodata in (
        (image_path, 2, "float32", np.ones((64, 64), dtype=np.float32), -9999),
        (label_path, 1, "uint8", np.zeros((64, 64), dtype=np.uint8), 255),
    ):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=64,
            height=64,
            count=count,
            dtype=dtype,
            crs="EPSG:32618",
            transform=transform,
            nodata=nodata,
        ) as raster:
            for band in range(1, count + 1):
                raster.write(data, band)

    result = write_spatial_patch_manifest(
        image_path,
        label_path,
        tmp_path / "patches.jsonl",
        patch_size=16,
        min_valid_fraction=1.0,
        train_fraction=0.5,
        validation_fraction=0.25,
    )

    assert result["split_counts"] == {"train": 4, "validation": 0, "test": 0}
    assert result["empty_splits"] == ["validation", "test"]
    assert result["splits_without_positive"] == ["train"]
