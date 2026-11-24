from copernicus_flood.ingestion.cems_client import CemsClient


def test_only_completed_feasible_products_are_download_candidates():
    activation = {
        "aois": [
            {
                "number": 1,
                "products": [
                    {
                        "id": 10,
                        "type": "DEL",
                        "feasible": True,
                        "version": {"number": 2, "statusCode": "F"},
                        "downloadPath": "https://rapidmapping.emergency.copernicus.eu/a.zip",
                    },
                    {
                        "id": 11,
                        "type": "GRA",
                        "feasible": False,
                        "version": {"number": 1, "statusCode": "N"},
                        "downloadPath": "https://rapidmapping.emergency.copernicus.eu/b.zip",
                    },
                ],
            }
        ]
    }

    products = CemsClient.iter_products(activation)

    assert [item["product"]["id"] for item in products] == [10]
    assert CemsClient.product_filename(
        products[0]["aoi"], products[0]["product"]
    ) == "EMSR865_AOI01_DEL_v2_id10.zip"
