from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from copernicus_flood.ingestion.cems_client import CemsClient
from copernicus_flood.ingestion.assets import download_verified_assets
from copernicus_flood.ingestion.cdse_auth import CdseOidcTokenProvider
from copernicus_flood.ingestion.cdse_client import CdseStacClient
from copernicus_flood.ingestion.matching import verify_product_candidates
from copernicus_flood.ingestion.registry import (
    CachedHttpClient,
    IngestionError,
    IngestionRegistry,
)
from copernicus_flood.ingestion.s3_assets import (
    CdseS3Client,
    download_verified_s3_assets,
)


def load_settings(path: Path) -> dict[str, Any]:
    try:
        settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise IngestionError(f"Impossibile leggere la configurazione: {path}") from error
    if not isinstance(settings, dict):
        raise IngestionError("La configurazione YAML deve contenere una mappa.")
    if settings.get("activation_code") != "EMSR865":
        raise IngestionError("La configurazione deve limitare l'attivazione a EMSR865.")
    return settings


def run_ingestion(
    settings: dict[str, Any],
    *,
    refresh: bool,
    download_verified: bool = False,
    asset_transport: str = "oidc",
) -> dict[str, Any]:
    cdse_settings = settings.get("cdse", {})
    if download_verified and not cdse_settings.get(
        "discover_sentinel_candidates", True
    ):
        raise IngestionError(
            "Il download asset richiede la ricerca e verifica dei candidati STAC."
        )
    if asset_transport not in {"oidc", "s3"}:
        raise IngestionError("Trasporto asset non supportato: usare oidc oppure s3.")
    token_provider = (
        CdseOidcTokenProvider.from_environment()
        if download_verified and asset_transport == "oidc"
        else None
    )

    data_root = Path(settings.get("data_root", "data"))
    registry_path = data_root / "manifests" / "ingestion.sqlite3"
    http_settings = settings.get("http", {})
    registry = IngestionRegistry(registry_path)
    http = CachedHttpClient(
        registry,
        timeout_seconds=int(http_settings.get("timeout_seconds", 60)),
        retries=int(http_settings.get("retries", 3)),
    )

    cems_settings = settings.get("cems", {})
    cems = CemsClient(http, cems_settings["api_root"])
    activation, activation_sha = cems.fetch_activation(
        settings["activation_code"], refresh=refresh
    )
    activation_path = (
        data_root / "raw" / "cems" / "EMSR865" / "activation.json"
    )
    cems.save_activation(activation, activation_path)

    products: list[dict[str, Any]] = []
    if cems_settings.get("download_products", True):
        products = cems.download_products(activation, data_root, refresh=refresh)

    stac_candidates: list[dict[str, Any]] = []
    asset_downloads: list[dict[str, Any]] = []
    if cdse_settings.get("discover_sentinel_candidates", True):
        cdse = CdseStacClient(http, cdse_settings["stac_root"])
        stac_candidates = cdse.discover_candidates(activation, refresh=refresh)
        for product in products:
            if product["type"] == "DEL":
                stac_candidates = verify_product_candidates(
                    Path(product["path"]), stac_candidates
                )
        candidate_path = (
            data_root / "manifests" / "emsr865_stac_candidates.json"
        )
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(
            json.dumps(stac_candidates, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if download_verified:
            if asset_transport == "oidc":
                if token_provider is None:
                    raise IngestionError("Provider token OIDC CDSE non disponibile.")
                asset_downloads = download_verified_assets(
                    stac_candidates,
                    http,
                    token_provider,
                    data_root,
                    refresh=refresh,
                )
            else:
                asset_downloads = download_verified_s3_assets(
                    stac_candidates,
                    CdseS3Client.from_environment(),
                    registry,
                    data_root,
                    refresh=refresh,
                )

    return {
        "activation_code": activation["code"],
        "activation_name": activation.get("name"),
        "activation_sha256": activation_sha,
        "activation_metadata": str(activation_path),
        "products": products,
        "stac_candidate_count": len(stac_candidates),
        "stac_candidates_verified": sum(
            candidate.get("verified") is True for candidate in stac_candidates
        ),
        "sentinel_asset_downloads": asset_downloads,
        "sentinel_asset_transport": asset_transport if download_verified else None,
        "registry": str(registry_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Acquisizione limitata a EMSR865 e ricerca di candidati Sentinel-1."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/settings.yaml"),
        help="Percorso della configurazione YAML.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Aggiorna le risposte API in cache e riscarica gli archivi CEMS.",
    )
    parser.add_argument(
        "--download-verified-assets",
        action="store_true",
        help="Scarica tutti gli asset VV/VH verificati (circa 6,2 GB per EMSR865).",
    )
    parser.add_argument(
        "--asset-transport",
        choices=("oidc", "s3"),
        default="oidc",
        help="Trasporto degli asset Sentinel-1; s3 usa le chiavi S3 CDSE.",
    )
    args = parser.parse_args()

    try:
        result = run_ingestion(
            load_settings(args.config),
            refresh=args.refresh,
            download_verified=args.download_verified_assets,
            asset_transport=args.asset_transport,
        )
    except (IngestionError, OSError, KeyError, ValueError) as error:
        print(f"ERRORE: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
