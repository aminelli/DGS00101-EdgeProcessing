from typing import Any

from copernicus_flood.ingestion.cdse_client import CdseStacClient


class _HttpStub:
    def __init__(self, features: list[dict[str, Any]]) -> None:
        self.features = features

    def get_json(self, url: str, *, params=None, refresh=False):
        return {"features": self.features}, None


def _item(item_id: str, datetime_value: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        },
        "properties": {
            "datetime": datetime_value,
            "sar:polarizations": ["VV", "VH"],
            "product:type": "IW_GRDH_1S",
            "platform": "sentinel-1a",
        },
        "assets": {
            "vv": {
                "href": "s3://eodata/vv.tif",
                "alternate": {
                    "https": {
                        "href": "https://zipper.dataspace.copernicus.eu/vv.tif",
                        "auth:refs": ["oidc"],
                    }
                },
            },
            "vh": {"href": "https://example.test/vh.tif"},
        },
    }


def test_stac_candidates_must_match_cems_time_within_tolerance():
    client = CdseStacClient(
        _HttpStub(
            [
                _item("near", "2026-02-02T23:15:00Z"),
                _item("outside", "2026-02-02T23:15:31Z"),
            ]
        ),
        "https://stac.dataspace.copernicus.eu/v1",
    )
    activation = {
        "code": "EMSR865",
        "aois": [
            {
                "name": "AOI01",
                "extent": "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
                "products": [
                    {
                        "id": "p1",
                        "type": "DEL",
                        "version": {"number": 2},
                        "images": [
                            {
                                "uuid": "image-1",
                                "fileName": "source.tif",
                                "acquisitionTime": "2026-02-02T23:14:00",
                            }
                        ],
                    }
                ],
            }
        ],
    }

    candidates = client.discover_candidates(activation)

    assert [candidate["stac_item_id"] for candidate in candidates] == ["near"]
    assert candidates[0]["time_delta_seconds"] == 60
    assert candidates[0]["verified"] is False
    assert candidates[0]["stac_assets"]["vv"].startswith(
        "https://zipper.dataspace.copernicus.eu/"
    )
    assert candidates[0]["stac_asset_authentication"]["vv"] == ["oidc"]
