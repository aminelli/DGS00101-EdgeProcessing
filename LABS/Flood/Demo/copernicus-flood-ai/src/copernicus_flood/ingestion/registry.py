from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class IngestionError(RuntimeError):
    """Errore esplicito durante richieste, cache o download."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_https_url(url: str, allowed_hosts: set[str]) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
        raise IngestionError(f"URL non consentito per l'ingestion: {url}")
    if parsed.username or parsed.password:
        raise IngestionError("Le credenziali non sono ammesse negli URL.")


class IngestionRegistry:
    """Cache delle risposte e registro locale delle richieste e dei download."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS http_cache (
                    url TEXT PRIMARY KEY,
                    body BLOB NOT NULL,
                    sha256 TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    etag TEXT,
                    last_modified TEXT,
                    fetched_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS request_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL,
                    requested_at TEXT NOT NULL,
                    cache_hit INTEGER NOT NULL,
                    status_code INTEGER,
                    response_sha256 TEXT
                );
                CREATE TABLE IF NOT EXISTS downloads (
                    url TEXT PRIMARY KEY,
                    local_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    etag TEXT,
                    completed_at TEXT NOT NULL
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    def cached_response(self, url: str) -> bytes | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT body, sha256 FROM http_cache WHERE url = ?", (url,)
            ).fetchone()
        if row is None:
            return None
        body, expected_hash = row
        if _sha256(body) != expected_hash:
            raise IngestionError(f"Cache HTTP corrotta per {url}")
        return body

    def store_response(self, url: str, response: requests.Response) -> str:
        body = response.content
        digest = _sha256(body)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO http_cache
                    (url, body, sha256, status_code, etag, last_modified, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    body=excluded.body,
                    sha256=excluded.sha256,
                    status_code=excluded.status_code,
                    etag=excluded.etag,
                    last_modified=excluded.last_modified,
                    fetched_at=excluded.fetched_at
                """,
                (
                    url,
                    body,
                    digest,
                    response.status_code,
                    response.headers.get("ETag"),
                    response.headers.get("Last-Modified"),
                    _utc_now(),
                ),
            )
        return digest

    def log_request(
        self,
        url: str,
        *,
        cache_hit: bool,
        status_code: int | None,
        response_sha256: str | None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO request_log
                    (url, requested_at, cache_hit, status_code, response_sha256)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    url,
                    _utc_now(),
                    int(cache_hit),
                    status_code,
                    response_sha256,
                ),
            )

    def completed_download(self, url: str) -> tuple[Path, str, int] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT local_path, sha256, size_bytes FROM downloads WHERE url = ?",
                (url,),
            ).fetchone()
        if row is None:
            return None
        return Path(row[0]), row[1], row[2]

    def store_download(
        self,
        url: str,
        path: Path,
        digest: str,
        size_bytes: int,
        etag: str | None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO downloads
                    (url, local_path, sha256, size_bytes, etag, completed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    local_path=excluded.local_path,
                    sha256=excluded.sha256,
                    size_bytes=excluded.size_bytes,
                    etag=excluded.etag,
                    completed_at=excluded.completed_at
                """,
                (url, str(path.resolve()), digest, size_bytes, etag, _utc_now()),
            )


class CachedHttpClient:
    def __init__(
        self,
        registry: IngestionRegistry,
        *,
        timeout_seconds: int = 60,
        retries: int = 3,
        session: requests.Session | None = None,
    ) -> None:
        self.registry = registry
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        retry = Retry(
            total=retries,
            connect=retries,
            read=retries,
            status=retries,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        refresh: bool = False,
    ) -> tuple[Any, str]:
        prepared = requests.Request("GET", url, params=params).prepare()
        request_url = prepared.url
        if not request_url:
            raise IngestionError(f"Impossibile costruire la richiesta per {url}")

        if not refresh:
            cached = self.registry.cached_response(request_url)
            if cached is not None:
                try:
                    payload = json.loads(cached)
                except json.JSONDecodeError as error:
                    raise IngestionError(
                        f"Risposta JSON non valida nella cache per {request_url}"
                    ) from error
                digest = _sha256(cached)
                self.registry.log_request(
                    request_url,
                    cache_hit=True,
                    status_code=200,
                    response_sha256=digest,
                )
                return payload, digest

        try:
            response = self.session.get(
                url, params=params, timeout=self.timeout_seconds
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.JSONDecodeError as error:
            self.registry.log_request(
                request_url,
                cache_hit=False,
                status_code=response.status_code,
                response_sha256=None,
            )
            raise IngestionError(f"La risposta non è JSON valido: {request_url}") from error
        except requests.RequestException as error:
            self.registry.log_request(
                request_url,
                cache_hit=False,
                status_code=getattr(getattr(error, "response", None), "status_code", None),
                response_sha256=None,
            )
            raise IngestionError(f"Richiesta HTTP non riuscita: {request_url}") from error

        digest = self.registry.store_response(request_url, response)
        self.registry.log_request(
            request_url,
            cache_hit=False,
            status_code=response.status_code,
            response_sha256=digest,
        )
        return payload, digest

    def download(
        self,
        url: str,
        destination: Path,
        *,
        allowed_hosts: set[str],
        headers: dict[str, str] | None = None,
        expected_size: int | None = None,
        refresh: bool = False,
    ) -> tuple[Path, str, bool]:
        validate_https_url(url, allowed_hosts)
        if expected_size is not None and expected_size <= 0:
            raise IngestionError("La dimensione attesa del download deve essere positiva.")
        destination.parent.mkdir(parents=True, exist_ok=True)

        record = self.registry.completed_download(url)
        if record and not refresh:
            saved_path, expected_hash, recorded_size = record
            if expected_size is not None and expected_size != recorded_size:
                raise IngestionError(
                    f"Dimensione registrata diversa dai metadati STAC: {url}"
                )
            if saved_path.is_file() and saved_path.stat().st_size == recorded_size:
                digest = _hash_file(saved_path)
                if digest == expected_hash:
                    return saved_path, digest, False

        if destination.exists() and not record and not refresh:
            raise IngestionError(
                f"File esistente ma non registrato; non sovrascrivo: {destination}"
            )

        partial = destination.with_name(destination.name + ".part")
        offset = partial.stat().st_size if partial.exists() else 0
        request_headers = dict(headers or {})
        if offset:
            request_headers["Range"] = f"bytes={offset}-"

        response: requests.Response | None = None
        try:
            response = self.session.get(
                url,
                headers=request_headers,
                timeout=self.timeout_seconds,
                stream=True,
            )
            if response.status_code == 416 and offset:
                response.close()
                partial.unlink(missing_ok=True)
                offset = 0
                response = self.session.get(
                    url,
                    headers=dict(headers or {}),
                    timeout=self.timeout_seconds,
                    stream=True,
                )
            response.raise_for_status()

            if offset and response.status_code == 206:
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {offset}-"):
                    raise IngestionError(
                        f"Content-Range non coerente per il download {url}"
                    )
                mode = "ab"
            else:
                offset = 0
                mode = "wb"

            response_bytes = 0
            with response:
                with partial.open(mode) as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
                            response_bytes += len(chunk)

            expected_response_size = response.headers.get("Content-Length")
            if expected_response_size is not None:
                expected_total = offset + int(expected_response_size)
                if partial.stat().st_size != expected_total:
                    raise IngestionError(
                        f"Download incompleto: ricevuti {partial.stat().st_size} "
                        f"byte, attesi {expected_total}: {url}"
                    )
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise IngestionError(
                    f"Dimensione download diversa dai metadati STAC: "
                    f"ricevuti {partial.stat().st_size}, attesi {expected_size}: {url}"
                )
        except requests.RequestException as error:
            if response is not None:
                response.close()
            raise IngestionError(
                f"Download HTTP non riuscito; eventuale .part conservato: {url}"
            ) from error
        except IngestionError:
            if response is not None:
                response.close()
            raise

        if response_bytes == 0:
            raise IngestionError(f"Il download non ha restituito dati: {url}")

        os.replace(partial, destination)
        digest = _hash_file(destination)
        self.registry.store_download(
            url,
            destination,
            digest,
            destination.stat().st_size,
            response.headers.get("ETag"),
        )
        return destination, digest, True


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
