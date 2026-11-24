from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject

from copernicus_flood.ingestion.registry import IngestionError


RASTER_NODATA = -9999.0


def build_vv_vh_stack(
    vv_path: Path | list[Path],
    vh_path: Path | list[Path],
    output_path: Path,
    *,
    bounds: tuple[float, float, float, float] | None = None,
    target_crs: CRS | str | None = None,
) -> Path:
    """Crea un mosaico VV/VH grezzo sulla griglia mosaicata della banda VV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vv_paths = [vv_path] if isinstance(vv_path, Path) else vv_path
    vh_paths = [vh_path] if isinstance(vh_path, Path) else vh_path
    if not vv_paths or len(vv_paths) != len(vh_paths):
        raise IngestionError("Fornire lo stesso numero di raster VV e VH.")

    with contextlib.ExitStack() as stack:
        vv_sources = [stack.enter_context(rasterio.open(path)) for path in vv_paths]
        vh_sources = [stack.enter_context(rasterio.open(path)) for path in vh_paths]
        source_datasets = vv_sources + vh_sources
        if any(source.count < 1 for source in source_datasets):
            raise IngestionError("Ogni raster Sentinel-1 deve contenere almeno una banda.")

        source_crs: list[CRS] = []
        for source in source_datasets:
            gcps, gcp_crs = source.gcps
            dataset_crs = source.crs or (gcp_crs if gcps else None)
            if dataset_crs is None:
                raise IngestionError(
                    "I raster VV e VH devono avere CRS o GCP georeferenziati."
                )
            source_crs.append(dataset_crs)

        output_crs = CRS.from_user_input(target_crs or source_crs[0])
        aligned_sources = []
        for source, src_crs in zip(source_datasets, source_crs, strict=True):
            warp = source.crs is None or src_crs != output_crs
            if warp:
                aligned_sources.append(
                    stack.enter_context(
                        WarpedVRT(
                            source,
                            crs=output_crs,
                            src_crs=src_crs,
                            nodata=source.nodata,
                            resampling=Resampling.bilinear,
                        )
                    )
                )
            else:
                aligned_sources.append(source)
        vv_sources = aligned_sources[: len(vv_paths)]
        vh_sources = aligned_sources[len(vv_paths) :]
        vv_crs = output_crs

        if bounds is not None and (
            len(bounds) != 4
            or bounds[0] >= bounds[2]
            or bounds[1] >= bounds[3]
        ):
            raise IngestionError("Limiti di mosaico non validi.")
        vv_mosaic, vv_transform = merge(
            vv_sources,
            indexes=1,
            nodata=RASTER_NODATA,
            dtype="float32",
            bounds=bounds,
            resampling=Resampling.bilinear,
        )
        vh_mosaic, vh_transform = merge(
            vh_sources,
            indexes=1,
            nodata=RASTER_NODATA,
            dtype="float32",
            bounds=bounds,
            res=vv_sources[0].res,
            resampling=Resampling.bilinear,
        )
        height, width = vv_mosaic.shape[1:]
        vv_values = vv_mosaic[0]
        vv_valid = np.isfinite(vv_values) & (vv_values != RASTER_NODATA)
        vh_values = np.full((height, width), RASTER_NODATA, dtype=np.float32)
        reproject(
            source=vh_mosaic[0],
            destination=vh_values,
            src_transform=vh_transform,
            src_crs=vv_crs,
            src_nodata=RASTER_NODATA,
            dst_transform=vv_transform,
            dst_crs=vv_crs,
            dst_nodata=RASTER_NODATA,
            resampling=Resampling.bilinear,
        )
        vh_valid_source = (vh_mosaic[0] != RASTER_NODATA).astype(np.uint8)
        vh_valid = np.zeros((height, width), dtype=np.uint8)
        reproject(
            source=vh_valid_source,
            destination=vh_valid,
            src_transform=vh_transform,
            src_crs=vv_crs,
            src_nodata=0,
            dst_transform=vv_transform,
            dst_crs=vv_crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
        valid = vv_valid & (vh_valid > 0) & np.isfinite(vh_values)
        vv_values = vv_values.copy()
        vv_values[~valid] = RASTER_NODATA
        vh_values[~valid] = RASTER_NODATA

        profile = vv_sources[0].profile.copy()
        profile.update(
            driver="GTiff",
            crs=output_crs,
            width=width,
            height=height,
            transform=vv_transform,
            count=2,
            dtype="float32",
            nodata=RASTER_NODATA,
            compress="deflate",
            tiled=False,
        )
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        with rasterio.open(output_path, "w", **profile) as output:
            output.write(vv_values, 1)
            output.write(vh_values, 2)
            output.write_mask(valid.astype(np.uint8) * 255)
            output.set_band_description(1, "VV_amplitude_raw")
            output.set_band_description(2, "VH_amplitude_raw")

    return output_path
