from __future__ import annotations

import getpass
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import BotoCoreError, ClientError

from .registry import IngestionError, IngestionRegistry, _hash_file


S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"
S3_BUCKET = "eodata"
TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=64 * 1024 * 1024,
    multipart_chunksize=64 * 1024 * 1024,
    max_concurrency=4,
)


class CdseS3Client:
    def __init__(self, access_key: str, secret_key: str) -> None:
        self.client = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="default",
        )

    @classmethod
    def from_environment(cls) -> CdseS3Client:
        access_key = os.environ.get("CDSE_S3_ACCESS_KEY")
        secret_key = os.environ.get("CDSE_S3_SECRET_KEY")
        if (not access_key or not secret_key) and not sys.stdin.isatty():
            raise IngestionError(
                "Impostare CDSE_S3_ACCESS_KEY e CDSE_S3_SECRET_KEY, oppure "
                "eseguire il comando in un terminale interattivo."
            )
        if not access_key:
            access_key = getpass.getpass("CDSE S3 access key: ")
        if not secret_key:
            secret_key = getpass.getpass("CDSE S3 secret key: ")
        if not access_key or not secret_key:
            raise IngestionError("Credenziali S3 CDSE non configurate.")
        return cls(access_key, secret_key)


def _parse_s3_href(href: str, item_id: str) -> tuple[str, str]:
    parsed = urlsplit(href)
    key = unquote(parsed.path.lstrip("/"))
    if (
        parsed.scheme != "s3"
        or parsed.netloc != S3_BUCKET
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
        or not key
    ):
        raise IngestionError(f"URI S3 CDSE non valida per l'item {item_id}.")
    return parsed.netloc, key


def download_verified_s3_assets(
    candidates: list[dict[str, Any]],
    s3: CdseS3Client,
    registry: IngestionRegistry,
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

        assets = candidate.get("stac_assets_s3")
        authentication = candidate.get("stac_asset_s3_authentication")
        sizes = candidate.get("stac_asset_sizes_bytes")
        if not isinstance(assets, dict) or not isinstance(authentication, dict):
            raise IngestionError(f"Asset S3 VV/VH assenti per l'item {item_id}.")

        for polarization in ("vv", "vh"):
            href = assets.get(polarization)
            auth_refs = authentication.get(polarization)
            expected_size = sizes.get(polarization) if isinstance(sizes, dict) else None
            if (
                not isinstance(href, str)
                or not isinstance(auth_refs, list)
                or "s3" not in auth_refs
            ):
                raise IngestionError(
                    f"Asset S3 {polarization.upper()} incompleto per l'item {item_id}."
                )
            if not isinstance(expected_size, int) or expected_size <= 0:
                raise IngestionError(
                    f"Dimensione STAC mancante per {polarization.upper()} di {item_id}."
                )
            bucket, key = _parse_s3_href(href, item_id)

            destination = (
                data_root
                / "raw"
                / "sentinel1"
                / "EMSR865"
                / f"product_{product_id}"
                / item_id
                / f"{polarization.upper()}.tif"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            record = registry.completed_download(href)
            if record and not refresh:
                saved_path, expected_hash, recorded_size = record
                if expected_size != recorded_size:
                    raise IngestionError(
                        f"Dimensione registrata diversa dai metadati STAC: {href}"
                    )
                if (
                    saved_path.is_file()
                    and saved_path.stat().st_size == recorded_size
                    and _hash_file(saved_path) == expected_hash
                ):
                    downloads.append(
                        {
                            "product_id": product_id,
                            "stac_item_id": item_id,
                            "polarization": polarization.upper(),
                            "path": str(saved_path),
                            "size_bytes": recorded_size,
                            "sha256": expected_hash,
                            "transferred": False,
                        }
                    )
                    continue
            if destination.exists() and not record and not refresh:
                raise IngestionError(
                    f"File esistente ma non registrato; non sovrascrivo: {destination}"
                )

            partial = destination.with_name(destination.name + ".part")
            try:
                metadata = s3.client.head_object(Bucket=bucket, Key=key)
                remote_size = metadata.get("ContentLength")
                if remote_size != expected_size:
                    raise IngestionError(
                        f"Dimensione S3 diversa dai metadati STAC per "
                        f"{polarization.upper()} di {item_id}."
                    )
                s3.client.download_file(
                    bucket,
                    key,
                    str(partial),
                    Config=TRANSFER_CONFIG,
                )
            except (BotoCoreError, ClientError) as error:
                raise IngestionError(
                    f"Download S3 CDSE non riuscito per "
                    f"{polarization.upper()} dell'item {item_id}."
                ) from error

            if not partial.is_file() or partial.stat().st_size != expected_size:
                raise IngestionError(
                    f"Download S3 incompleto per {polarization.upper()} "
                    f"dell'item {item_id}."
                )
            os.replace(partial, destination)
            digest = _hash_file(destination)
            registry.store_download(
                href,
                destination,
                digest,
                destination.stat().st_size,
                metadata.get("ETag"),
            )
            downloads.append(
                {
                    "product_id": product_id,
                    "stac_item_id": item_id,
                    "polarization": polarization.upper(),
                    "path": str(destination),
                    "size_bytes": destination.stat().st_size,
                    "sha256": digest,
                    "transferred": True,
                }
            )
    if not downloads:
        raise IngestionError("Il manifesto non contiene asset verificati da scaricare.")
    return downloads
