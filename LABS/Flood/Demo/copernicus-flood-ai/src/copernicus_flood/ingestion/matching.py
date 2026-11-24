from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

import geopandas as gpd
from shapely.geometry import shape

from .registry import IngestionError


MAX_TIME_DELTA_SECONDS = 90
MIN_SCENE_COVERAGE = 0.9
MIN_SOURCE_COVERAGE = 0.98


def verify_candidate_group(
    source_geometry: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    source_id: int,
    source_datetime: str,
) -> list[dict[str, Any]]:
    source = gpd.GeoDataFrame(
        geometry=[shape(source_geometry)], crs="EPSG:4326"
    )
    target_crs = source.estimate_utm_crs()
    if target_crs is None:
        raise IngestionError("Impossibile stimare il CRS metrico per il footprint.")
    source_geometry_projected = source.to_crs(target_crs).geometry.iloc[0]
    source_datetime_parsed = datetime.fromisoformat(
        source_datetime.replace("Z", "+00:00")
    ).astimezone(timezone.utc)

    projected_candidates = []
    for candidate in candidates:
        geometry = candidate.get("stac_geometry")
        datetime_value = candidate.get("stac_datetime")
        if not isinstance(geometry, dict) or not isinstance(datetime_value, str):
            continue
        candidate_datetime = datetime.fromisoformat(
            datetime_value.replace("Z", "+00:00")
        ).astimezone(timezone.utc)
        if (
            abs((candidate_datetime - source_datetime_parsed).total_seconds())
            > MAX_TIME_DELTA_SECONDS
        ):
            continue
        candidate_geometry = gpd.GeoDataFrame(
            geometry=[shape(geometry)], crs="EPSG:4326"
        ).to_crs(target_crs).geometry.iloc[0]
        if candidate_geometry.area == 0:
            continue
        scene_coverage = (
            source_geometry_projected.intersection(candidate_geometry).area
            / candidate_geometry.area
        )
        if scene_coverage >= MIN_SCENE_COVERAGE:
            projected_candidates.append((candidate, candidate_geometry, scene_coverage))

    if not projected_candidates:
        return candidates

    source_coverage = (
        source_geometry_projected.intersection(
            gpd.GeoSeries(
                [geometry for _, geometry, _ in projected_candidates],
                crs=target_crs,
            ).union_all()
        ).area
        / source_geometry_projected.area
    )
    if source_coverage < MIN_SOURCE_COVERAGE:
        return candidates

    verified_ids = {id(candidate) for candidate, _, _ in projected_candidates}
    for candidate in candidates:
        if id(candidate) not in verified_ids:
            continue
        scene_coverage = next(
            coverage
            for matched_candidate, _, coverage in projected_candidates
            if matched_candidate is candidate
        )
        candidate.update(
            {
                "status": "verified",
                "verified": True,
                "verification_method": (
                    "CEMS Sentinel-1 source ID, UTC acquisition time, and "
                    "image-footprint coverage"
                ),
                "cems_source_id": source_id,
                "cems_source_datetime": source_datetime,
                "scene_footprint_coverage": scene_coverage,
                "source_footprint_union_coverage": source_coverage,
            }
        )
    return candidates


def verify_product_candidates(
    product_zip: Path, candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    product_match = re.search(r"_id(\d+)\.zip$", product_zip.name, re.IGNORECASE)
    if product_match is None:
        raise IngestionError(f"ID prodotto non riconoscibile: {product_zip.name}")
    product_id = int(product_match.group(1))
    product_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("product_id") == product_id
    ]
    if not product_candidates:
        return candidates

    try:
        with ZipFile(product_zip) as archive:
            source_tables = [
                name
                for name in archive.namelist()
                if "_source_" in Path(name).name.lower()
                and name.lower().endswith(".dbf")
            ]
            footprint_layers = [
                name
                for name in archive.namelist()
                if "imagefootprinta" in Path(name).name.lower()
                and name.lower().endswith(".json")
            ]
            if len(source_tables) != 1 or len(footprint_layers) != 1:
                raise IngestionError(
                    f"Layer source/footprint non univoci in {product_zip.name}."
                )
            footprints = json.loads(archive.read(footprint_layers[0]))
    except (OSError, BadZipFile, json.JSONDecodeError) as error:
        raise IngestionError(
            f"Impossibile leggere i layer source/footprint: {product_zip}"
        ) from error

    try:
        source_table = gpd.read_file(
            f"zip://{product_zip.resolve()}!{source_tables[0]}"
        )
    except (OSError, ValueError) as error:
        raise IngestionError(f"Tabella source non leggibile: {product_zip}") from error
    if not {"source_nam", "src_date", "source_tm", "src_id"}.issubset(
        source_table.columns
    ):
        raise IngestionError("Campi richiesti assenti nella tabella CEMS source.")
    sentinel_sources = source_table[
        source_table["source_nam"].astype(str).str.strip() == "Sentinel-1"
    ]
    if len(sentinel_sources) != 1:
        raise IngestionError(
            f"Attesa una sola fonte Sentinel-1 in {product_zip.name}, "
            f"trovate {len(sentinel_sources)}."
        )
    source_record = sentinel_sources.iloc[0]
    try:
        source_date = datetime.strptime(
            source_record["src_date"], "%d/%m/%Y"
        ).date()
        source_clock = datetime.strptime(source_record["source_tm"], "T%H:%M:%SZ").time()
        source_datetime = datetime.combine(
            source_date, source_clock, tzinfo=timezone.utc
        ).isoformat()
        source_id = int(source_record["src_id"])
    except (TypeError, ValueError) as error:
        raise IngestionError(
            f"Data, ora o ID Sentinel-1 non validi in {product_zip.name}."
        ) from error

    matching_features = [
        feature
        for feature in footprints.get("features", [])
        if isinstance(feature, dict)
        and isinstance(feature.get("properties"), dict)
        and feature["properties"].get("or_src_id") == source_id
    ]
    if len(matching_features) != 1:
        raise IngestionError(
            f"Atteso un footprint per source ID {source_id} in {product_zip.name}, "
            f"trovati {len(matching_features)}."
        )

    verify_candidate_group(
        matching_features[0]["geometry"],
        product_candidates,
        source_id=source_id,
        source_datetime=source_datetime,
    )
    for candidate in product_candidates:
        candidate.pop("stac_geometry", None)
    return candidates
