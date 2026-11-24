from __future__ import annotations

import os
import time
from typing import Any

import requests

from .registry import IngestionError


TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)


class CdseOidcTokenProvider:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        *,
        session: requests.Session | None = None,
        timeout_seconds: int = 60,
    ) -> None:
        if not client_id or not client_secret:
            raise IngestionError("Credenziali OIDC CDSE non configurate.")
        self.client_id = client_id
        self.client_secret = client_secret
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds
        self._access_token: str | None = None
        self._expires_at = 0.0

    @classmethod
    def from_environment(
        cls, *, timeout_seconds: int = 60
    ) -> CdseOidcTokenProvider:
        client_id = os.environ.get("CDSE_CLIENT_ID")
        client_secret = os.environ.get("CDSE_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise IngestionError(
                "Impostare CDSE_CLIENT_ID e CDSE_CLIENT_SECRET nell'ambiente "
                "locale per scaricare gli asset autenticati."
            )
        return cls(
            client_id,
            client_secret,
            timeout_seconds=timeout_seconds,
        )

    def authorization_headers(self) -> dict[str, str]:
        if self._access_token and time.monotonic() < self._expires_at:
            return {"Authorization": f"Bearer {self._access_token}"}

        try:
            response = self.session.post(
                TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload: Any = response.json()
        except requests.RequestException as error:
            raise IngestionError("Richiesta token OIDC CDSE non riuscita.") from error
        except ValueError as error:
            raise IngestionError("Risposta token OIDC CDSE non valida.") from error

        if not isinstance(payload, dict):
            raise IngestionError("Risposta token OIDC CDSE non valida.")
        access_token = payload.get("access_token")
        expires_in = payload.get("expires_in")
        if (
            not isinstance(access_token, str)
            or not access_token
            or not isinstance(expires_in, (int, float))
            or expires_in <= 0
        ):
            raise IngestionError(
                "La risposta OIDC CDSE non contiene un token o una scadenza validi."
            )

        self._access_token = access_token
        self._expires_at = time.monotonic() + max(0, expires_in - 60)
        return {"Authorization": f"Bearer {access_token}"}
