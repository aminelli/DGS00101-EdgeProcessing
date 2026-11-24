from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .registry import CachedHttpClient, IngestionError


ACTIVATIONS_PATH = "/backend/dashboard-api/public-activations/"
ALLOWED_DOWNLOAD_HOSTS = {
    "rapidmapping.emergency.copernicus.eu",
    "rapidmapping-viewer.s3.eu-west-1.amazonaws.com",
}


class CemsClient:
    def __init__(self, http: CachedHttpClient, api_root: str) -> None:
        self.http = http
        self.api_root = api_root.rstrip("/")

    def fetch_activation(self, activation_code: str, *, refresh: bool = False) -> tuple[dict[str, Any], str]:
        if activation_code != "EMSR865":
            raise IngestionError("Questa demo è limitata all'attivazione EMSR865.")

        payload, digest = self.http.get_json(
            f"{self.api_root}{ACTIVATIONS_PATH}",
            params={"code": activation_code},
            refresh=refresh,
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise IngestionError("Risposta CEMS priva della lista results attesa.")

        matches = [
            item
            for item in payload["results"]
            if isinstance(item, dict) and item.get("code") == activation_code
        ]
        if len(matches) != 1:
            raise IngestionError(
                f"Attesa una sola attivazione {activation_code}, trovate {len(matches)}."
            )
        return matches[0], digest

    @staticmethod
    def iter_products(activation: dict[str, Any]) -> list[dict[str, Any]]:
        products: list[dict[str, Any]] = []
        aois = activation.get("aois")
        if not isinstance(aois, list):
            raise IngestionError("La risposta CEMS non contiene una lista aois valida.")

        for aoi in aois:
            if not isinstance(aoi, dict) or not isinstance(aoi.get("products"), list):
                continue
            for product in aoi["products"]:
                if not isinstance(product, dict):
                    continue
                version = product.get("version")
                if (
                    product.get("type") in {"DEL", "GRA"}
                    and product.get("feasible") is True
                    and isinstance(version, dict)
                    and version.get("statusCode") == "F"
                    and isinstance(product.get("downloadPath"), str)
                ):
                    products.append({"aoi": aoi, "product": product})
        return products

    @staticmethod
    def product_filename(aoi: dict[str, Any], product: dict[str, Any]) -> str:
        aoi_number = aoi.get("number")
        product_id = product.get("id")
        product_type = product.get("type")
        version = product.get("version", {}).get("number")
        monitoring_number = product.get("monitoringNumber", 0)
        if not all(value is not None for value in (aoi_number, product_id, product_type, version)):
            raise IngestionError("Metadati insufficienti per nominare l'archivio CEMS.")
        if not re.fullmatch(r"[A-Z0-9_-]+", str(product_type)):
            raise IngestionError(f"Tipo prodotto CEMS non valido: {product_type}")
        suffix = f"_MONIT{int(monitoring_number):02d}" if product.get("monitoring") else ""
        return (
            f"EMSR865_AOI{int(aoi_number):02d}_{product_type}{suffix}"
            f"_v{int(version)}_id{int(product_id)}.zip"
        )

    def download_products(
        self,
        activation: dict[str, Any],
        data_root: Path,
        *,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        downloaded: list[dict[str, Any]] = []
        for item in self.iter_products(activation):
            aoi = item["aoi"]
            product = item["product"]
            url = product["downloadPath"]
            parsed = urlsplit(url)
            if parsed.hostname not in ALLOWED_DOWNLOAD_HOSTS:
                raise IngestionError(f"Host download CEMS non previsto: {url}")

            destination = (
                data_root
                / "raw"
                / "cems"
                / "EMSR865"
                / f"AOI{int(aoi['number']):02d}"
                / self.product_filename(aoi, product)
            )
            path, digest, transferred = self.http.download(
                url,
                destination,
                allowed_hosts=ALLOWED_DOWNLOAD_HOSTS,
                refresh=refresh,
            )
            downloaded.append(
                {
                    "product_id": product.get("id"),
                    "type": product.get("type"),
                    "version": product.get("version", {}).get("number"),
                    "path": str(path),
                    "sha256": digest,
                    "transferred": transferred,
                }
            )
        return downloaded

    @staticmethod
    def save_activation(activation: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(activation, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
