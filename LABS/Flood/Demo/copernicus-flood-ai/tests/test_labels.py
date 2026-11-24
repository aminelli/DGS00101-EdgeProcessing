import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import rasterio
from rasterio.transform import from_origin, rowcol

from copernicus_flood.preprocessing.labels import build_label_mask


def _write_geojson(archive: ZipFile, name: str, features: list[dict]):
    archive.writestr(
        name,
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {
                    "type": "name",
                    "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
                },
                "features": features,
            }
        ),
    )


def _feature(coords, notation=None):
    properties = {}
    if notation is not None:
        properties["notation"] = notation
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {"type": "Polygon", "coordinates": [coords]},
    }


def test_label_mask_encodes_background_flood_modelled_and_outside(tmp_path: Path):
    product_zip = tmp_path / "EMSR865_AOI01_DEL_v2_id1.zip"
    aoi_ring = [
        [-76.0, 9.0],
        [-75.99, 9.0],
        [-75.99, 8.99],
        [-76.0, 8.99],
        [-76.0, 9.0],
    ]
    flood_ring = [
        [-75.997, 8.998],
        [-75.993, 8.998],
        [-75.993, 8.994],
        [-75.997, 8.994],
        [-75.997, 8.998],
    ]
    modelled_ring = [
        [-75.999, 8.998],
        [-75.998, 8.998],
        [-75.998, 8.996],
        [-75.999, 8.996],
        [-75.999, 8.998],
    ]
    with ZipFile(product_zip, "w") as archive:
        _write_geojson(
            archive, "EMSR865_areaOfInterestA_v2.json", [_feature(aoi_ring)]
        )
        _write_geojson(
            archive,
            "EMSR865_observedEventA_v2.json",
            [_feature(flood_ring, "Flooded area")],
        )
        _write_geojson(
            archive,
            "EMSR865_modelledEventA_v2.json",
            [_feature(modelled_ring, "Flooded area")],
        )
    reference = tmp_path / "reference.tif"
    transform = from_origin(-76.005, 9.005, 0.0001, 0.0001)
    with rasterio.open(
        reference,
        "w",
        driver="GTiff",
        width=100,
        height=100,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
        nodata=0,
    ) as raster:
        raster.write(np.ones((100, 100), dtype=np.uint8), 1)

    output = tmp_path / "labels.tif"
    result = build_label_mask(reference, product_zip, output)

    with rasterio.open(output) as mask:
        values = mask.read(1)
        modelled_row, modelled_col = rowcol(transform, -75.9985, 8.997)
        assert values[modelled_row, modelled_col] == 255
    assert 1 in values
    assert 0 in values
    assert 255 in values
    assert result["negative_label_assumption"] == "aoi_background_is_negative"


def test_label_mask_allows_missing_optional_modelled_layer(tmp_path: Path):
    product_zip = tmp_path / "EMSR865_AOI01_DEL_v2_id1.zip"
    aoi_ring = [
        [-76.0, 9.0],
        [-75.99, 9.0],
        [-75.99, 8.99],
        [-76.0, 8.99],
        [-76.0, 9.0],
    ]
    flood_ring = [
        [-75.999, 8.998],
        [-75.998, 8.998],
        [-75.998, 8.996],
        [-75.999, 8.996],
        [-75.999, 8.998],
    ]
    with ZipFile(product_zip, "w") as archive:
        _write_geojson(
            archive, "EMSR865_areaOfInterestA_v2.json", [_feature(aoi_ring)]
        )
        _write_geojson(
            archive,
            "EMSR865_observedEventA_v2.json",
            [_feature(flood_ring, "Flooded area")],
        )

    reference = tmp_path / "reference.tif"
    transform = from_origin(-76.005, 9.005, 0.0001, 0.0001)
    with rasterio.open(
        reference,
        "w",
        driver="GTiff",
        width=100,
        height=100,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
        nodata=0,
    ) as raster:
        raster.write(np.ones((100, 100), dtype=np.uint8), 1)

    result = build_label_mask(reference, product_zip, tmp_path / "labels.tif")

    assert result["flood_pixel_count"] > 0
