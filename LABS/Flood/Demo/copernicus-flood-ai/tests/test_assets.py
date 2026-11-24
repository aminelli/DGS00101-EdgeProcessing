from pathlib import Path

from copernicus_flood.ingestion.assets import download_verified_assets


class _HttpStub:
    def __init__(self):
        self.calls = []

    def download(self, url, destination, **kwargs):
        self.calls.append((url, destination, kwargs))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"asset")
        return destination, "sha256", True


class _TokenStub:
    def authorization_headers(self):
        return {"Authorization": "Bearer token"}


def test_only_verified_assets_are_downloaded(tmp_path: Path):
    candidates = [
        {
            "verified": True,
            "product_id": 2435,
            "stac_item_id": "verified-item",
            "stac_assets": {
                "vv": "https://zipper.dataspace.copernicus.eu/vv",
                "vh": "https://zipper.dataspace.copernicus.eu/vh",
            },
            "stac_asset_authentication": {
                "vv": ["oidc"],
                "vh": ["oidc"],
            },
            "stac_asset_sizes_bytes": {"vv": 5, "vh": 5},
        },
        {
            "verified": False,
            "product_id": 2435,
            "stac_item_id": "unverified-item",
        },
    ]
    http = _HttpStub()

    downloads = download_verified_assets(
        candidates, http, _TokenStub(), tmp_path
    )

    assert [item["polarization"] for item in downloads] == ["VV", "VH"]
    assert len(http.calls) == 2
    assert all(
        call[2]["headers"] == {"Authorization": "Bearer token"}
        and call[2]["expected_size"] == 5
        for call in http.calls
    )
