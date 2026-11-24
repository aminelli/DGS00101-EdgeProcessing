import hashlib
from pathlib import Path

import pytest

from copernicus_flood.ingestion.registry import (
    CachedHttpClient,
    IngestionError,
    IngestionRegistry,
)


class FakeResponse:
    status_code = 200
    headers = {"ETag": '"abc"'}
    content = b'{"ok": true}'

    def raise_for_status(self):
        return None

    def json(self):
        return {"ok": True}


class FakeSession:
    def __init__(self):
        self.calls = 0

    def mount(self, *_args):
        return None

    def get(self, *_args, **_kwargs):
        self.calls += 1
        return FakeResponse()


class FakeDownloadResponse:
    status_code = 200
    headers = {"ETag": '"file-etag"', "Content-Length": "7"}

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        assert chunk_size > 0
        yield b"product"

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def close(self):
        return None


class FakeDownloadSession(FakeSession):
    def __init__(self):
        super().__init__()
        self.request_headers = None

    def get(self, *_args, **_kwargs):
        self.calls += 1
        self.request_headers = _kwargs.get("headers")
        return FakeDownloadResponse()


def test_json_response_is_cached_and_call_is_tracked(tmp_path: Path):
    session = FakeSession()
    registry = IngestionRegistry(tmp_path / "ledger.sqlite3")
    client = CachedHttpClient(registry, session=session)

    first, first_hash = client.get_json("https://example.test/data")
    second, second_hash = client.get_json("https://example.test/data")

    assert first == second == {"ok": True}
    assert first_hash == second_hash == hashlib.sha256(b'{"ok": true}').hexdigest()
    assert session.calls == 1

    with registry._connect() as connection:
        rows = connection.execute(
            "SELECT cache_hit FROM request_log ORDER BY id"
        ).fetchall()
    assert rows == [(0,), (1,)]


def test_completed_download_is_not_fetched_again(tmp_path: Path):
    session = FakeDownloadSession()
    registry = IngestionRegistry(tmp_path / "ledger.sqlite3")
    client = CachedHttpClient(registry, session=session)
    target = tmp_path / "product.zip"

    first_path, first_hash, first_transferred = client.download(
        "https://rapidmapping.emergency.copernicus.eu/product.zip",
        target,
        allowed_hosts={"rapidmapping.emergency.copernicus.eu"},
        headers={"Authorization": "Bearer test-token"},
        expected_size=7,
    )
    second_path, second_hash, second_transferred = client.download(
        "https://rapidmapping.emergency.copernicus.eu/product.zip",
        target,
        allowed_hosts={"rapidmapping.emergency.copernicus.eu"},
    )

    assert first_path == second_path == target
    assert first_hash == second_hash == hashlib.sha256(b"product").hexdigest()
    assert first_transferred is True
    assert second_transferred is False
    assert session.calls == 1
    assert session.request_headers == {"Authorization": "Bearer test-token"}


def test_download_rejects_size_mismatch_before_registering_file(tmp_path: Path):
    session = FakeDownloadSession()
    registry = IngestionRegistry(tmp_path / "ledger.sqlite3")
    client = CachedHttpClient(registry, session=session)
    target = tmp_path / "product.zip"

    with pytest.raises(IngestionError, match="Dimensione download diversa"):
        client.download(
            "https://rapidmapping.emergency.copernicus.eu/product.zip",
            target,
            allowed_hosts={"rapidmapping.emergency.copernicus.eu"},
            expected_size=8,
        )

    assert not target.exists()
    assert (tmp_path / "product.zip.part").read_bytes() == b"product"
    assert registry.completed_download(
        "https://rapidmapping.emergency.copernicus.eu/product.zip"
    ) is None
