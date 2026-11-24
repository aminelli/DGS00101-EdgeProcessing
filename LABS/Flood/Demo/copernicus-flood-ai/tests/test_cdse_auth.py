from copernicus_flood.ingestion.cdse_auth import CdseOidcTokenProvider


class _TokenResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"access_token": "temporary-token", "expires_in": 3600}


class _TokenSession:
    def __init__(self):
        self.calls = []

    def post(self, url, *, data, timeout):
        self.calls.append((url, data, timeout))
        return _TokenResponse()


def test_oidc_token_is_cached_in_memory():
    session = _TokenSession()
    provider = CdseOidcTokenProvider(
        "local-client-id",
        "local-client-secret",
        session=session,
    )

    first = provider.authorization_headers()
    second = provider.authorization_headers()

    assert first == second == {"Authorization": "Bearer temporary-token"}
    assert len(session.calls) == 1
    assert session.calls[0][1]["grant_type"] == "client_credentials"
