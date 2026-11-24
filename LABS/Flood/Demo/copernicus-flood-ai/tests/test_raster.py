from pathlib import Path

import numpy as np
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from copernicus_flood.preprocessing.raster import build_vv_vh_stack


def _write_raster(path: Path, data: np.ndarray, transform, crs="EPSG:32618"):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=0,
    ) as raster:
        raster.write(data, 1)


def test_vv_vh_stack_preserves_values_and_aligns_bands(tmp_path: Path):
    vv = tmp_path / "vv.tif"
    vh = tmp_path / "vh.tif"
    output = tmp_path / "stack.tif"
    transform = from_origin(500000, 1000000, 10, 10)
    _write_raster(vv, np.full((8, 8), 100, dtype=np.uint16), transform)
    _write_raster(vh, np.full((8, 8), 50, dtype=np.uint16), transform)

    build_vv_vh_stack(vv, vh, output)

    with rasterio.open(output) as stack:
        assert stack.count == 2
        assert stack.descriptions == ("VV_amplitude_raw", "VH_amplitude_raw")
        assert np.all(stack.read(1) == 100)
        assert np.all(stack.read(2) == 50)
        assert stack.dataset_mask().all()


def test_vv_vh_stack_warps_sentinel_gcps_to_requested_crs(tmp_path: Path):
    vv = tmp_path / "vv_gcps.tif"
    vh = tmp_path / "vh_gcps.tif"
    output = tmp_path / "stack_gcps.tif"
    gcps = [
        GroundControlPoint(row=0, col=0, x=-75.95, y=9.05),
        GroundControlPoint(row=0, col=7, x=-75.94, y=9.05),
        GroundControlPoint(row=7, col=0, x=-75.95, y=9.04),
        GroundControlPoint(row=7, col=7, x=-75.94, y=9.04),
    ]
    for path, value in ((vv, 100), (vh, 50)):
        _write_raster(
            path,
            np.full((8, 8), value, dtype=np.uint16),
            from_origin(1, 1, 1, 1),
        )
        with rasterio.open(path, "r+") as raster:
            raster.gcps = gcps, CRS.from_epsg(4326)

    target_crs = CRS.from_epsg(32618)
    bounds = transform_bounds(
        "EPSG:4326", target_crs, -75.951, 9.039, -75.939, 9.051
    )
    build_vv_vh_stack(
        vv,
        vh,
        output,
        bounds=bounds,
        target_crs=target_crs,
    )

    with rasterio.open(output) as stack:
        assert stack.crs == target_crs
        assert stack.dataset_mask().any()
        assert np.any(stack.read(1) == 100)
        assert np.any(stack.read(2) == 50)
