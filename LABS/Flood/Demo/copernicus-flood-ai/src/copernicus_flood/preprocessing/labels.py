from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import mapping

from copernicus_flood.ingestion.registry import IngestionError


IGNORE_VALUE = 255
FLOOD_VALUE = 1
BACKGROUND_VALUE = 0


def read_product_layer(
    product_zip: Path, layer_name: str, *, required: bool = True
) -> gpd.GeoDataFrame:
    """Legge un singolo layer GeoJSON dall'archivio EMSR senza estrarlo su disco."""
    suffix = f"_{layer_name}_"
    try:
        with ZipFile(product_zip) as archive:
            matches = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".json")
                and suffix.lower() in Path(name).name.lower()
            ]
            if not matches and not required:
                return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            if len(matches) != 1:
                raise IngestionError(
                    f"Atteso un solo layer {layer_name} in {product_zip.name}, "
                    f"trovati {len(matches)}."
                )
            collection = json.loads(archive.read(matches[0]))
    except (OSError, BadZipFile, json.JSONDecodeError) as error:
        raise IngestionError(f"Archivio CEMS non leggibile: {product_zip}") from error

    features = collection.get("features") if isinstance(collection, dict) else None
    if not isinstance(features, list):
        raise IngestionError(f"GeoJSON non valido per il layer {layer_name}.")
    if not features:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    frame = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty]
    if frame.empty:
        raise IngestionError(f"Il layer {layer_name} non contiene geometrie valide.")
    return frame


def _rasterize_union(
    geometries: list[Any],
    *,
    out_shape: tuple[int, int],
    transform: Any,
) -> np.ndarray:
    valid_geometries = [geometry for geometry in geometries if geometry is not None and not geometry.is_empty]
    if not valid_geometries:
        return np.zeros(out_shape, dtype=np.uint8)
    return rasterize(
        ((mapping(geometry), 1) for geometry in valid_geometries),
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )


def _valid_flood_geometries(frame: gpd.GeoDataFrame, layer_name: str) -> list[Any]:
    if "notation" not in frame.columns:
        raise IngestionError(f"Attributo `notation` assente nel layer {layer_name}.")
    selected = frame[frame["notation"] == "Flooded area"]
    if selected.empty:
        raise IngestionError(f"Nessun poligono Flooded area nel layer {layer_name}.")
    return list(selected.geometry)


def build_label_mask(
    reference_raster: Path,
    product_zip: Path,
    output_path: Path,
    *,
    source_layers: tuple[str, ...] = ("modelledEventA", "maximumFloodExtentA"),
) -> dict[str, Any]:
    """Crea una maschera 0/1/255 sulla griglia di un raster Sentinel-1."""
    aoi = read_product_layer(product_zip, "areaOfInterestA")
    observed = read_product_layer(product_zip, "observedEventA")
    modelled_layers = [
        read_product_layer(product_zip, layer_name, required=False)
        for layer_name in source_layers
    ]

    with rasterio.open(reference_raster) as reference:
        if reference.crs is None:
            raise IngestionError("Il raster di riferimento non ha un CRS definito.")
        if aoi.empty:
            raise IngestionError("Il layer areaOfInterestA non contiene geometrie.")

        aoi = aoi.to_crs(reference.crs)
        observed = observed.to_crs(reference.crs)
        modelled_layers = [
            layer.to_crs(reference.crs) for layer in modelled_layers
        ]
        shape = (reference.height, reference.width)
        valid_input = reference.dataset_mask() > 0

        aoi_mask = _rasterize_union(
            list(aoi.geometry), out_shape=shape, transform=reference.transform
        ).astype(bool)
        if not np.any(aoi_mask & valid_input):
            raise IngestionError("L'AOI non interseca pixel validi del raster.")
        observed_mask = _rasterize_union(
            _valid_flood_geometries(observed, "observedEventA"),
            out_shape=shape,
            transform=reference.transform,
        ).astype(bool)
        modelled_mask = np.zeros(shape, dtype=np.uint8)
        for layer in modelled_layers:
            if layer.empty:
                continue
            if "notation" not in layer.columns:
                raise IngestionError(
                    "Attributo `notation` assente in un layer modellato."
                )
            flood_features = layer[layer["notation"].astype(str).str.contains(
                "flood", case=False, na=False
            )]
            if not flood_features.empty:
                modelled_mask |= _rasterize_union(
                    list(flood_features.geometry),
                    out_shape=shape,
                    transform=reference.transform,
                )

        valid = aoi_mask & valid_input
        if not np.any(observed_mask & valid):
            raise IngestionError(
                "I poligoni Flooded area osservati non intersecano pixel validi "
                "del raster."
            )
        labels = np.full(shape, IGNORE_VALUE, dtype=np.uint8)
        labels[valid] = BACKGROUND_VALUE
        labels[observed_mask & valid] = FLOOD_VALUE
        labels[(modelled_mask > 0) & ~observed_mask & valid] = IGNORE_VALUE

        output_path.parent.mkdir(parents=True, exist_ok=True)
        profile = reference.profile.copy()
        profile.update(
            count=1,
            dtype="uint8",
            nodata=IGNORE_VALUE,
            compress="deflate",
            tiled=False,
        )
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        with rasterio.open(output_path, "w", **profile) as output:
            output.write(labels, 1)
            output.set_band_description(
                1, "flood=1, assumed_background=0, unknown_or_nodata=255"
            )

    return {
        "output": str(output_path),
        "classes": {
            "flood": FLOOD_VALUE,
            "assumed_background": BACKGROUND_VALUE,
            "ignore": IGNORE_VALUE,
        },
        "negative_label_assumption": "aoi_background_is_negative",
        "observed_layer": "observedEventA",
        "modelled_layers_ignored": list(source_layers),
        "valid_pixel_count": int(np.count_nonzero(labels != IGNORE_VALUE)),
        "flood_pixel_count": int(np.count_nonzero(labels == FLOOD_VALUE)),
        "assumption": (
            "Il background interno all'AOI è trattato come negativo per uso didattico; "
            "non è una classe non-allagata verificata da CEMS."
        ),
    }


def infer_product_version(product_zip: Path) -> int:
    match = re.search(r"_v(\d+)_id\d+\.zip$", product_zip.name, flags=re.IGNORECASE)
    if not match:
        raise IngestionError(
            f"Il nome archivio non contiene una versione prodotto riconoscibile: "
            f"{product_zip.name}"
        )
    return int(match.group(1))
