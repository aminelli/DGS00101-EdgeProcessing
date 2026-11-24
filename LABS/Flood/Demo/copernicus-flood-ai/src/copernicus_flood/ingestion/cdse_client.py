from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from urllib.parse import urlsplit

from shapely import wkt
from shapely.errors import GEOSException
from shapely.geometry import shape

from .registry import CachedHttpClient, IngestionError


STAC_HOST = "stac.dataspace.copernicus.eu"
ACQUISITION_TIME_TOLERANCE_SECONDS = 90


def _parse_utc_datetime(value: str, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise IngestionError(f"Data/ora {label} non valida: {value}") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class CdseStacClient:
    def __init__(self, http: CachedHttpClient, stac_root: str) -> None:
        self.http = http
        self.stac_root = stac_root.rstrip("/")
        if urlsplit(self.stac_root).hostname != STAC_HOST:
            raise IngestionError("Endpoint STAC non previsto per questa demo.")

    def _search_day(
        self,
        aoi_wkt: str,
        acquisition_date: date,
        *,
        refresh: bool,
    ) -> list[dict[str, Any]]:
        try:
            aoi = wkt.loads(aoi_wkt)
        except (GEOSException, TypeError, ValueError) as error:
            raise IngestionError("Geometria AOI CEMS non interpretabile come WKT.") from error
        if aoi.is_empty or not aoi.is_valid:
            raise IngestionError("Geometria AOI CEMS vuota o non valida.")

        start = datetime.combine(acquisition_date, time.min, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        query = {
            "collections": "sentinel-1-grd",
            "bbox": ",".join(f"{coordinate:.8f}" for coordinate in aoi.bounds),
            "datetime": f"{start.isoformat()}/{end.isoformat()}",
            "limit": 100,
        }
        search_url = f"{self.stac_root}/search"
        payload, _ = self.http.get_json(search_url, params=query, refresh=refresh)
        if not isinstance(payload, dict) or not isinstance(payload.get("features"), list):
            raise IngestionError("Risposta STAC priva della lista features attesa.")

        items = [item for item in payload["features"] if isinstance(item, dict)]
        for link in payload.get("links", []):
            if not isinstance(link, dict) or link.get("rel") != "next":
                continue
            next_url = link.get("href")
            if not isinstance(next_url, str) or urlsplit(next_url).hostname != STAC_HOST:
                raise IngestionError("Link di paginazione STAC non consentito.")
            seen = {search_url}
            while next_url:
                if next_url in seen:
                    raise IngestionError("Ciclo rilevato nella paginazione STAC.")
                seen.add(next_url)
                page, _ = self.http.get_json(next_url, refresh=refresh)
                if not isinstance(page, dict) or not isinstance(page.get("features"), list):
                    raise IngestionError("Pagina STAC successiva non valida.")
                items.extend(item for item in page["features"] if isinstance(item, dict))
                next_links = [
                    item.get("href")
                    for item in page.get("links", [])
                    if isinstance(item, dict) and item.get("rel") == "next"
                ]
                if len(next_links) > 1:
                    raise IngestionError("Risposta STAC con link next ambigui.")
                next_url = next_links[0] if next_links else None
                if next_url and urlsplit(next_url).hostname != STAC_HOST:
                    raise IngestionError("Link di paginazione STAC non consentito.")
        return items

    def discover_candidates(
        self,
        activation: dict[str, Any],
        *,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        images_seen: set[tuple[str, str, str]] = set()

        for aoi in activation.get("aois", []):
            if not isinstance(aoi, dict) or not isinstance(aoi.get("extent"), str):
                continue
            try:
                aoi_geometry = wkt.loads(aoi["extent"])
            except (GEOSException, TypeError, ValueError) as error:
                raise IngestionError(
                    f"Geometria AOI non interpretabile: {aoi.get('name')}"
                ) from error
            for product in aoi.get("products", []):
                if not isinstance(product, dict):
                    continue
                if product.get("type") != "DEL":
                    continue
                for image in product.get("images", []):
                    if not isinstance(image, dict):
                        continue
                    acquired_at = image.get("acquisitionTime")
                    if not isinstance(acquired_at, str):
                        continue
                    cems_datetime = _parse_utc_datetime(
                        acquired_at, label="immagine CEMS"
                    )
                    acquisition_date = cems_datetime.date()

                    image_key = (
                        str(product.get("id")),
                        str(image.get("uuid")),
                        acquired_at,
                    )
                    if image_key in images_seen:
                        continue
                    images_seen.add(image_key)

                    for item in self._search_day(
                        aoi["extent"], acquisition_date, refresh=refresh
                    ):
                        geometry = item.get("geometry")
                        properties = item.get("properties")
                        assets = item.get("assets")
                        if not isinstance(geometry, dict) or not isinstance(properties, dict):
                            continue
                        try:
                            intersects = shape(geometry).intersects(aoi_geometry)
                        except (TypeError, ValueError):
                            continue

                        polarizations = properties.get("sar:polarizations")
                        stac_datetime = properties.get("datetime")
                        if not isinstance(stac_datetime, str):
                            continue
                        stac_acquired_at = _parse_utc_datetime(
                            stac_datetime, label="item STAC"
                        )
                        time_delta_seconds = abs(
                            (stac_acquired_at - cems_datetime).total_seconds()
                        )
                        if not isinstance(polarizations, list):
                            continue
                        product_type = str(properties.get("product:type", ""))
                        version = product.get("version")
                        if (
                            not intersects
                            or time_delta_seconds
                            > ACQUISITION_TIME_TOLERANCE_SECONDS
                            or not isinstance(assets, dict)
                            or "vv" not in assets
                            or "vh" not in assets
                            or not {"VV", "VH"}.issubset(set(polarizations))
                            or "GRDH" not in product_type
                        ):
                            continue
                        stac_assets: dict[str, str] = {}
                        stac_assets_s3: dict[str, str] = {}
                        asset_authentication: dict[str, list[str]] = {}
                        s3_asset_authentication: dict[str, list[str]] = {}
                        asset_sizes: dict[str, int] = {}
                        for polarization in ("vv", "vh"):
                            asset = assets.get(polarization)
                            if not isinstance(asset, dict):
                                continue
                            alternate = asset.get("alternate")
                            https_alternate = (
                                alternate.get("https")
                                if isinstance(alternate, dict)
                                else None
                            )
                            href = (
                                https_alternate.get("href")
                                if isinstance(https_alternate, dict)
                                else asset.get("href")
                            )
                            if isinstance(href, str):
                                stac_assets[polarization] = href
                            s3_alternate = (
                                alternate.get("s3")
                                if isinstance(alternate, dict)
                                else None
                            )
                            s3_href = (
                                s3_alternate.get("href")
                                if isinstance(s3_alternate, dict)
                                else asset.get("href")
                            )
                            if isinstance(s3_href, str) and s3_href.startswith(
                                "s3://"
                            ):
                                stac_assets_s3[polarization] = s3_href
                            s3_auth_refs = (
                                s3_alternate.get("auth:refs")
                                if isinstance(s3_alternate, dict)
                                else asset.get("auth:refs")
                            )
                            if isinstance(s3_auth_refs, list):
                                s3_asset_authentication[polarization] = s3_auth_refs
                            size = asset.get("file:size")
                            if isinstance(size, int) and size > 0:
                                asset_sizes[polarization] = size
                            auth_refs = (
                                https_alternate.get("auth:refs")
                                if isinstance(https_alternate, dict)
                                else asset.get("auth:refs")
                            )
                            if isinstance(auth_refs, list):
                                asset_authentication[polarization] = auth_refs
                        candidates.append(
                            {
                                "status": "candidate",
                                "verified": False,
                                "reason": (
                                    "Intersezione AOI, data, prodotto GRDH e asset VV/VH "
                                    "compatibili; manca un identificativo comune CEMS-CDSE."
                                ),
                                "activation_code": activation.get("code"),
                                "aoi_name": aoi.get("name"),
                                "product_id": product.get("id"),
                                "product_type": product.get("type"),
                                "product_version": (
                                    version.get("number")
                                    if isinstance(version, dict)
                                    else None
                                ),
                                "cems_image_uuid": image.get("uuid"),
                                "cems_file_name": image.get("fileName"),
                                "cems_acquisition_time": acquired_at,
                                "time_delta_seconds": time_delta_seconds,
                                "stac_item_id": item.get("id"),
                                "stac_datetime": stac_datetime,
                                "stac_geometry": geometry,
                                "stac_platform": properties.get("platform"),
                                "stac_product_type": product_type,
                                "stac_polarizations": polarizations,
                                "stac_assets": stac_assets,
                                "stac_assets_s3": stac_assets_s3,
                                "stac_asset_authentication": asset_authentication,
                                "stac_asset_s3_authentication": (
                                    s3_asset_authentication
                                ),
                                "stac_asset_sizes_bytes": asset_sizes,
                            }
                        )
        return candidates
