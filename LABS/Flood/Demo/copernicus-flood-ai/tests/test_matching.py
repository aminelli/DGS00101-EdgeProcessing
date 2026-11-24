from copernicus_flood.ingestion.matching import verify_candidate_group


def _polygon(west: float, east: float) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [west, 8.0],
                [east, 8.0],
                [east, 9.0],
                [west, 9.0],
                [west, 8.0],
            ]
        ],
    }


def test_verification_requires_scene_and_union_footprint_coverage():
    candidates = [
        {
            "stac_item_id": "left",
            "stac_datetime": "2026-02-02T23:14:30Z",
            "stac_geometry": _polygon(-76.0, -75.5),
        },
        {
            "stac_item_id": "right",
            "stac_datetime": "2026-02-02T23:15:00Z",
            "stac_geometry": _polygon(-75.5, -75.0),
        },
        {
            "stac_item_id": "outside",
            "stac_datetime": "2026-02-02T23:14:30Z",
            "stac_geometry": _polygon(-75.0, -74.5),
        },
    ]

    verified = verify_candidate_group(
        _polygon(-76.0, -75.0),
        candidates,
        source_id=2,
        source_datetime="2026-02-02T23:14:00+00:00",
    )

    assert [item.get("verified") for item in verified] == [True, True, None]
    assert all(
        item["source_footprint_union_coverage"] > 0.98 for item in verified[:2]
    )
