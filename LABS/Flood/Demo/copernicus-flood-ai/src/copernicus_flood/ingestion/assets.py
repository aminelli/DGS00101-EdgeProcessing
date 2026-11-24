from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .cdse_auth import CdseOidcTokenProvider
from .registry import CachedHttpClient, IngestionError


ASSET_HOSTS = {"zipper.dataspace.copernicus.eu"}


def download_verified_assets(
    candidates: list[dict[str, Any]],
    http: CachedHttpClient,
    token_provider: CdseOidcTokenProvider,
    data_root: Path,
    *,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    downloads: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.get("verified") is not True:
            continue
        item_id = candidate.get("stac_item_id")
        product_id = candidate.get("product_id")
        if (
            not isinstance(item_id, str)
            or not re.fullmatch(r"[A-Za-z0-9._-]+", item_id)
            or not isinstance(product_id, int)
        ):
            raise IngestionError("ID prodotto o item STAC non validi nel manifesto.")

        assets = candidate.get("stac_assets")
        authentication = candidate.get("stac_asset_authentication")
        sizes = candidate.get("stac_asset_sizes_bytes")
        if not isinstance(assets, dict) or not isinstance(authentication, dict):
            raise IngestionError(f"Asset VV/VH assenti per l'item {item_id}.")

        for polarization in ("vv", "vh"):
            href = assets.get(polarization)
            auth_refs = authentication.get(polarization)
            expected_size = sizes.get(polarization) if isinstance(sizes, dict) else None
            if not isinstance(href, str) or not isinstance(auth_refs, list):
                raise IngestionError(
                    f"Asset {polarization.upper()} incompleto per l'item {item_id}."
                )
            if "oidc" not in auth_refs:
                raise IngestionError(
                    f"Metodo di autenticazione OIDC assente per l'asset {item_id}."
                )
            if urlsplit(href).hostname not in ASSET_HOSTS:
                raise IngestionError(
                    f"Host HTTPS asset CDSE non previsto per l'item {item_id}."
                )
            if not isinstance(expected_size, int) or expected_size <= 0:
                raise IngestionError(
                    f"Dimensione STAC mancante per {polarization.upper()} di {item_id}."
                )

            destination = (
                data_root
                / "raw"
                / "sentinel1"
                / "EMSR865"
                / f"product_{product_id}"
                / item_id
                / f"{polarization.upper()}.tif"
            )
            path, digest, transferred = http.download(
                href,
                destination,
                allowed_hosts=ASSET_HOSTS,
                headers=token_provider.authorization_headers(),
                expected_size=expected_size,
                refresh=refresh,
            )
            downloads.append(
                {
                    "product_id": product_id,
                    "stac_item_id": item_id,
                    "polarization": polarization.upper(),
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": digest,
                    "transferred": transferred,
                }
            )
    if not downloads:
        raise IngestionError("Il manifesto non contiene asset verificati da scaricare.")
    return downloads
