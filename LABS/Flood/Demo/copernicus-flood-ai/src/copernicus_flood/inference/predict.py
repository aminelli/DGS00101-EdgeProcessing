from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.errors import RasterioError
import torch
from rasterio.features import geometry_mask, shapes
from rasterio.windows import Window
from shapely.geometry import mapping, shape

from copernicus_flood.ingestion.registry import IngestionError
from copernicus_flood.models.flood.unet import UNet


IGNORE_VALUE = 255
PROBABILITY_NODATA = -9999.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[UNet, dict[str, Any]]:
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise IngestionError(f"Checkpoint non leggibile: {checkpoint_path}") from error
    if not isinstance(checkpoint, dict) or checkpoint.get("activation_code") != "EMSR865":
        raise IngestionError("Checkpoint non valido o fuori perimetro EMSR865.")
    if checkpoint.get("negative_label_assumption") != "aoi_background_is_negative":
        raise IngestionError("Checkpoint privo dell'assunzione sulle etichette.")
    config = checkpoint.get("model_config")
    if not isinstance(config, dict) or config.get("architecture") != "unet":
        raise IngestionError("Architettura checkpoint non supportata.")
    model = UNet(
        in_channels=int(config["in_channels"]),
        num_classes=int(config["num_classes"]),
        base_channels=int(config["base_channels"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _write_preview(mask_path: Path, output_path: Path, max_dimension: int) -> None:
    with rasterio.open(mask_path) as source:
        scale = min(1.0, max_dimension / max(source.width, source.height))
        height = max(1, round(source.height * scale))
        width = max(1, round(source.width * scale))
        reduced = source.read(
            1,
            out_shape=(height, width),
            resampling=rasterio.enums.Resampling.nearest,
        )
    rgba = np.zeros((4, height, width), dtype=np.uint8)
    valid = reduced != IGNORE_VALUE
    background = reduced == 0
    flooded = reduced == 1
    rgba[0, background] = 238
    rgba[1, background] = 242
    rgba[2, background] = 245
    rgba[0, flooded] = 0
    rgba[1, flooded] = 110
    rgba[2, flooded] = 255
    rgba[3, valid] = 255
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="PNG",
        width=width,
        height=height,
        count=4,
        dtype="uint8",
    ) as output:
        output.write(rgba)
        output.set_band_description(1, "red")
        output.set_band_description(2, "green")
        output.set_band_description(3, "blue")
        output.set_band_description(4, "alpha")


def _polygonize_flood(
    mask_path: Path,
    geojson_path: Path,
    geopackage_path: Path,
    *,
    minimum_pixels: int,
) -> int:
    features: list[dict[str, Any]] = []
    with rasterio.open(mask_path) as source:
        if source.crs is None:
            raise IngestionError("La maschera di predizione non ha un CRS.")
        pixel_area = abs(
            source.transform.a * source.transform.e
            - source.transform.b * source.transform.d
        )
        if pixel_area <= 0:
            raise IngestionError("La trasformazione raster non definisce pixel validi.")
        block_size = 1024
        for row_offset in range(0, source.height, block_size):
            for col_offset in range(0, source.width, block_size):
                height = min(block_size, source.height - row_offset)
                width = min(block_size, source.width - col_offset)
                window = Window(col_offset, row_offset, width, height)
                values = source.read(1, window=window)
                flood = values == 1
                if not np.any(flood):
                    continue
                transform = source.window_transform(window)
                for geometry, value in shapes(
                    values,
                    mask=flood,
                    transform=transform,
                    connectivity=8,
                ):
                    polygon = shape(geometry)
                    pixel_count = int(round(polygon.area / pixel_area))
                    if pixel_count < minimum_pixels:
                        continue
                    features.append(
                        {
                            "type": "Feature",
                            "properties": {
                                "class": "flood",
                                "pixel_count": pixel_count,
                            },
                            "geometry": geometry,
                        }
                    )

        geojson_path.parent.mkdir(parents=True, exist_ok=True)
        if features:
            frame = gpd.GeoDataFrame.from_features(features, crs=source.crs)
            frame.to_file(geopackage_path, driver="GPKG", index=False)
            frame.to_crs("EPSG:4326").to_file(
                geojson_path, driver="GeoJSON", index=False
            )
        else:
            geojson_path.write_text(
                json.dumps(
                    {"type": "FeatureCollection", "features": []},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
    return len(features)


def predict_raster(args: argparse.Namespace) -> dict[str, Any]:
    if args.tile_size <= 0 or args.minimum_polygon_pixels <= 0:
        raise IngestionError("tile-size e minimum-polygon-pixels devono essere positivi.")
    if args.preview_max_dimension <= 0:
        raise IngestionError("preview-max-dimension deve essere positivo.")
    if not 0 < args.threshold < 1:
        raise IngestionError("La soglia probabilistica deve essere tra 0 e 1.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise IngestionError("CUDA è stato richiesto ma non è disponibile.")
    model, checkpoint = _load_model(args.checkpoint, device)
    means = np.asarray(checkpoint.get("channel_mean"), dtype=np.float32)
    standard_deviations = np.asarray(checkpoint.get("channel_std"), dtype=np.float32)
    if (
        means.shape != (2,)
        or standard_deviations.shape != (2,)
        or np.any(standard_deviations <= 0)
    ):
        raise IngestionError("Statistiche VV/VH non valide nel checkpoint.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = args.output_dir / "flood_mask.tif"
    probability_path = args.output_dir / "flood_probability.tif"
    with rasterio.open(args.input_stack) as source:
        if source.count != 2 or source.crs is None:
            raise IngestionError("Input atteso: stack VV/VH a due bande con CRS.")
        try:
            aoi_frame = gpd.read_file(args.aoi_vector)
        except (OSError, ValueError) as error:
            raise IngestionError(f"Vettore AOI non leggibile: {args.aoi_vector}") from error
        if aoi_frame.crs is None:
            raise IngestionError("Il vettore AOI non ha un CRS definito.")
        aoi_frame = aoi_frame[
            aoi_frame.geometry.notna() & ~aoi_frame.geometry.is_empty
        ].to_crs(source.crs)
        if aoi_frame.empty:
            raise IngestionError("Il vettore AOI non contiene geometrie.")
        aoi_geometries = [mapping(geometry) for geometry in aoi_frame.geometry]
        aoi_bounds = aoi_frame.total_bounds
        source_bounds = source.bounds
        if (
            aoi_bounds[0] >= source_bounds.right
            or aoi_bounds[2] <= source_bounds.left
            or aoi_bounds[1] >= source_bounds.top
            or aoi_bounds[3] <= source_bounds.bottom
        ):
            raise IngestionError("L'AOI non interseca lo stack Sentinel-1.")
        profile = source.profile.copy()
        mask_profile = profile.copy()
        mask_profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=IGNORE_VALUE,
            compress="deflate",
            tiled=False,
        )
        probability_profile = profile.copy()
        probability_profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=PROBABILITY_NODATA,
            compress="deflate",
            tiled=False,
        )
        mask_profile.pop("blockxsize", None)
        mask_profile.pop("blockysize", None)
        probability_profile.pop("blockxsize", None)
        probability_profile.pop("blockysize", None)
        with (
            rasterio.open(mask_path, "w", **mask_profile) as mask_output,
            rasterio.open(
                probability_path, "w", **probability_profile
            ) as probability_output,
        ):
            valid_pixel_count = 0
            flooded_pixel_count = 0
            for row_offset in range(0, source.height, args.tile_size):
                for col_offset in range(0, source.width, args.tile_size):
                    height = min(args.tile_size, source.height - row_offset)
                    width = min(args.tile_size, source.width - col_offset)
                    window = Window(col_offset, row_offset, width, height)
                    window_transform = source.window_transform(window)
                    raw = source.read((1, 2), window=window).astype(np.float32)
                    valid = (
                        (source.read_masks(1, window=window) > 0)
                        & (source.read_masks(2, window=window) > 0)
                        & np.isfinite(raw).all(axis=0)
                    )
                    valid &= geometry_mask(
                        aoi_geometries,
                        out_shape=(height, width),
                        transform=window_transform,
                        invert=True,
                    )
                    if source.nodata is not None:
                        valid &= np.all(raw != source.nodata, axis=0)
                    for channel in range(2):
                        raw[channel] = np.log1p(np.maximum(raw[channel], 0))
                        raw[channel][~valid] = 0
                    normalized = (
                        raw - means[:, None, None]
                    ) / standard_deviations[:, None, None]
                    padding = (
                        0,
                        args.tile_size - width,
                        0,
                        args.tile_size - height,
                    )
                    padded = torch.nn.functional.pad(
                        torch.from_numpy(normalized), padding, value=0
                    ).unsqueeze(0)
                    with torch.inference_mode():
                        logits = model(padded.to(device))
                        probability = torch.softmax(logits, dim=1)[0, 1]
                    probability_array = (
                        probability[:height, :width].cpu().numpy().astype(np.float32)
                    )
                    tile_mask = np.full(
                        (height, width), IGNORE_VALUE, dtype=np.uint8
                    )
                    tile_mask[valid] = (
                        probability_array[valid] >= args.threshold
                    ).astype(np.uint8)
                    valid_pixel_count += int(np.count_nonzero(valid))
                    flooded_pixel_count += int(
                        np.count_nonzero(tile_mask == 1)
                    )
                    probability_array[~valid] = PROBABILITY_NODATA
                    mask_output.write(tile_mask, 1, window=window)
                    probability_output.write(
                        probability_array, 1, window=window
                    )
            mask_output.set_band_description(1, "0=background, 1=flood, 255=ignore")
            probability_output.set_band_description(1, "flood_probability")
    if valid_pixel_count == 0:
        raise IngestionError("L'AOI non interseca pixel validi dello stack.")

    geojson_path = args.output_dir / "flood_polygons.geojson"
    geopackage_path = args.output_dir / "flood_polygons.gpkg"
    polygon_count = _polygonize_flood(
        mask_path,
        geojson_path,
        geopackage_path,
        minimum_pixels=args.minimum_polygon_pixels,
    )
    preview_path = args.output_dir / "flood_preview.png"
    _write_preview(mask_path, preview_path, args.preview_max_dimension)
    result = {
        "activation_code": "EMSR865",
        "input_stack": str(args.input_stack),
        "aoi_vector": str(args.aoi_vector),
        "input_stack_sha256": _sha256_file(args.input_stack),
        "model_checkpoint": str(args.checkpoint),
        "model_checkpoint_sha256": _sha256_file(args.checkpoint),
        "threshold": args.threshold,
        "minimum_polygon_pixels": args.minimum_polygon_pixels,
        "polygonization_method": (
            "Windowed 1024-pixel vectorization; adjoining window polygons are not dissolved."
        ),
        "mask_geotiff": str(mask_path),
        "probability_geotiff": str(probability_path),
        "geojson": str(geojson_path),
        "geopackage": str(geopackage_path) if polygon_count else None,
        "preview_png": str(preview_path),
        "flood_polygon_count": polygon_count,
        "valid_pixel_count": valid_pixel_count,
        "flood_pixel_count": flooded_pixel_count,
        "negative_label_assumption": checkpoint["negative_label_assumption"],
        "evaluation_scope": checkpoint["evaluation_scope"],
        "warning": (
            "La mappa è una predizione esplorativa del modello addestrato solo "
            "sull'attivazione EMSR865; non è una valutazione operativa."
        ),
    }
    (args.output_dir / "prediction_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inferenza U-Net Flood con output geospaziali EMSR865."
    )
    parser.add_argument("--input-stack", type=Path, required=True)
    parser.add_argument("--aoi-vector", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--minimum-polygon-pixels", type=int, default=4)
    parser.add_argument("--preview-max-dimension", type=int, default=2048)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    try:
        result = predict_raster(args)
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
