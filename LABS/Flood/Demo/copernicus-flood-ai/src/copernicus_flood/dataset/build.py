from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import rasterio
from rasterio.errors import RasterioError

from copernicus_flood.dataset.patches import write_spatial_patch_manifest
from copernicus_flood.ingestion.registry import IngestionError, IngestionRegistry
from copernicus_flood.preprocessing.labels import (
    build_label_mask,
    infer_product_version,
    read_product_layer,
)
from copernicus_flood.preprocessing.raster import build_vv_vh_stack


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocessing Sentinel-1 e maschere CEMS per EMSR865."
    )
    parser.add_argument("--product-zip", type=Path, required=True)
    parser.add_argument("--vv", type=Path, nargs="+", required=True)
    parser.add_argument("--vh", type=Path, nargs="+", required=True)
    parser.add_argument("--stac-item-ids", nargs="+", required=True)
    parser.add_argument(
        "--association-manifest",
        type=Path,
        default=Path("data/manifests/emsr865_stac_candidates.json"),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("data/manifests/ingestion.sqlite3"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--min-valid-fraction", type=float, default=0.8)
    args = parser.parse_args()

    try:
        args.product_zip = args.product_zip.resolve()
        args.vv = [path.resolve() for path in args.vv]
        args.vh = [path.resolve() for path in args.vh]
        args.output_dir = args.output_dir.resolve()
        args.association_manifest = args.association_manifest.resolve()
        args.registry = args.registry.resolve()
        args.data_root = args.data_root.resolve()
        version = infer_product_version(args.product_zip)
        if len(args.vv) != len(args.vh):
            raise IngestionError("Fornire VV e VH per ogni item Sentinel-1.")
        product_match = re.search(r"_id(\d+)\.zip$", args.product_zip.name, re.IGNORECASE)
        if product_match is None:
            raise IngestionError("ID prodotto non riconoscibile dal nome dello ZIP.")
        product_id = int(product_match.group(1))
        associations = json.loads(
            args.association_manifest.read_text(encoding="utf-8")
        )
        if not isinstance(associations, list):
            raise IngestionError("Il manifesto di associazione deve essere una lista.")
        verified = [
            item
            for item in associations
            if isinstance(item, dict)
            and item.get("product_id") == product_id
            and item.get("verified") is True
        ]
        if not verified:
            raise IngestionError(
                f"Nessuna scena Sentinel-1 verificata per il prodotto {product_id}."
            )
        verified_by_id = {item.get("stac_item_id"): item for item in verified}
        if len(verified_by_id) != len(verified):
            raise IngestionError("Il manifesto contiene ID item STAC duplicati.")
        if (
            len(args.stac_item_ids) != len(set(args.stac_item_ids))
            or set(args.stac_item_ids) != set(verified_by_id)
        ):
            raise IngestionError(
                "Specificare tutti e soli gli item verificati che coprono "
                f"il prodotto {product_id}: {', '.join(sorted(verified_by_id))}"
            )

        input_by_path = {
            path.resolve(): ("VV", path) for path in args.vv
        } | {
            path.resolve(): ("VH", path) for path in args.vh
        }
        if len(input_by_path) != len(args.vv) + len(args.vh):
            raise IngestionError("Sono stati passati percorsi raster duplicati.")
        registry = IngestionRegistry(args.registry)
        input_paths: dict[str, list[Path]] = {"VV": [], "VH": []}
        provenance: list[dict[str, object]] = []
        expected_paths: set[Path] = set()
        for item_id in sorted(verified_by_id):
            item = verified_by_id[item_id]
            asset_urls = item.get("stac_assets")
            asset_s3_urls = item.get("stac_assets_s3")
            asset_sizes = item.get("stac_asset_sizes_bytes")
            if (
                not isinstance(asset_urls, dict)
                or not isinstance(asset_sizes, dict)
            ):
                raise IngestionError(f"Asset metadata mancanti per {item_id}.")
            for polarization in ("VV", "VH"):
                key = polarization.lower()
                url = asset_urls.get(key)
                s3_urls = asset_s3_urls if isinstance(asset_s3_urls, dict) else {}
                s3_url = s3_urls.get(key)
                size = asset_sizes.get(key)
                if not isinstance(url, str) or not isinstance(size, int):
                    raise IngestionError(
                        f"Metadati {polarization} mancanti per {item_id}."
                    )
                if isinstance(s3_url, str) and registry.completed_download(s3_url):
                    url = s3_url
                expected_path = (
                    args.data_root
                    / "raw"
                    / "sentinel1"
                    / "EMSR865"
                    / f"product_{product_id}"
                    / item_id
                    / f"{polarization}.tif"
                ).resolve()
                expected_paths.add(expected_path)
                record = registry.completed_download(url)
                provided = input_by_path.get(expected_path)
                if provided is None or provided[0] != polarization:
                    raise IngestionError(
                        f"File {polarization} non corrispondente all'asset "
                        f"verificato {item_id}: {expected_path}"
                    )
                path = provided[1]
                if (
                    record is None
                    or record[0].resolve() != expected_path
                    or record[2] != size
                    or not path.is_file()
                    or path.stat().st_size != size
                ):
                    raise IngestionError(
                        f"Download {polarization} assente o incoerente nel registro: "
                        f"{path}"
                    )
                digest = hashlib.sha256()
                with path.open("rb") as raster_file:
                    for chunk in iter(lambda: raster_file.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != record[1]:
                    raise IngestionError(f"Checksum non valido per {path}.")
                input_paths[polarization].append(path)
                provenance.append(
                    {
                        "stac_item_id": item_id,
                        "product_id": product_id,
                        "polarization": polarization,
                        "path": str(path),
                        "size_bytes": size,
                        "sha256": record[1],
                        "asset_href": url,
                        "acquisition_time": item.get("stac_datetime"),
                        "platform": item.get("stac_platform"),
                        "polarizations": item.get("stac_polarizations"),
                        "source_id": item.get("cems_source_id"),
                        "source_time": item.get("cems_source_datetime"),
                        "scene_footprint_coverage": item.get(
                            "scene_footprint_coverage"
                        ),
                        "source_footprint_union_coverage": item.get(
                            "source_footprint_union_coverage"
                        ),
                    }
                )
        if set(input_by_path) != expected_paths:
            raise IngestionError(
                "I raster forniti non corrispondono esattamente agli item STAC "
                "verificati nel manifesto."
            )

        stack_path = args.output_dir / "sentinel1_vv_vh_stack.tif"
        label_path = args.output_dir / f"emsr865_observed_flood_v{version}.tif"
        patch_manifest = args.output_dir / "patches.jsonl"

        aoi = read_product_layer(args.product_zip, "areaOfInterestA")
        target_crs = aoi.estimate_utm_crs()
        if target_crs is None:
            raise IngestionError("Impossibile stimare il CRS metrico per l'AOI.")
        aoi_bounds = tuple(
            float(value) for value in aoi.to_crs(target_crs).total_bounds
        )
        build_vv_vh_stack(
            input_paths["VV"],
            input_paths["VH"],
            stack_path,
            bounds=aoi_bounds,
            target_crs=target_crs,
        )
        label_info = build_label_mask(stack_path, args.product_zip, label_path)
        patch_info = write_spatial_patch_manifest(
            stack_path,
            label_path,
            patch_manifest,
            patch_size=args.patch_size,
            min_valid_fraction=args.min_valid_fraction,
        )

        result = {
            "dataset_status": "preprocessing_only_invalid_splits",
            "activation_code": "EMSR865",
            "product_version": version,
            "scene_association": {
                "status": "verified",
                "activation_code": "EMSR865",
                "product_id": product_id,
                "stac_item_ids": sorted(verified_by_id),
                "method": "verified CEMS source ID, UTC time, and source-footprint coverage",
            },
            "input_stack": str(stack_path),
            "patch_manifest": patch_manifest.name,
            "label_mask": label_info,
            "patches": patch_info,
            "input_assets": provenance,
            "negative_label_assumption": "aoi_background_is_negative",
            "evaluation_scope": "spatial_holdout_within_single_activation_EMSR865",
            "input_preprocessing": (
                "VV/VH conservate come ampiezza grezza float32; "
                "nessuna calibrazione radiometrica applicata; il training usa "
                "log1p e statistiche calcolate esclusivamente sullo split train."
            ),
        }
        all_splits_have_both_classes = all(
            patch_info["split_counts"][split] > 0
            and patch_info["class_pixel_counts"][split]["positive"] > 0
            and patch_info["class_pixel_counts"][split]["negative"] > 0
            for split in ("train", "validation", "test")
        )
        if all_splits_have_both_classes:
            result["dataset_status"] = "ready_for_training"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "dataset_manifest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except (
        IngestionError,
        OSError,
        ValueError,
        RasterioError,
        json.JSONDecodeError,
    ) as error:
        print(f"ERRORE: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
