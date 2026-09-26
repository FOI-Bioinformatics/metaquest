"""A fake HTTP server at urllib3's connection-pool level, for tests of retrying sessions.

``serve`` replaces ``HTTPConnectionPool._make_request`` (the call that would open a socket), so a
``requests.Session`` with a mounted ``Retry`` goes through urllib3's real retry logic while every
response comes from a handler function. Retry backoff sleeps are skipped. No network is used.
"""

import io
from typing import Callable, List, Tuple
from urllib.parse import urlsplit

from urllib3.connectionpool import HTTPConnectionPool
from urllib3.response import HTTPResponse
from urllib3.util.retry import Retry


def serve(monkeypatch, handler: Callable[[str], Tuple[int, bytes]]) -> List[str]:
    """Answer every request with ``handler(path)``, a (status, body) pair; return the paths requested.

    ``path`` is the request path without the query string.
    """
    requested: List[str] = []

    def fake_make_request(self, conn, method, url, *args, **kwargs):
        path = urlsplit(url).path
        requested.append(path)
        status, body = handler(path)
        return HTTPResponse(
            body=io.BytesIO(body),
            status=status,
            headers={"Content-Type": "application/json"},
            preload_content=False,
            decode_content=False,
            request_method=method,
            request_url=url,
        )

    monkeypatch.setattr(HTTPConnectionPool, "_make_request", fake_make_request)
    monkeypatch.setattr(Retry, "sleep", lambda self, response=None: None)
    return requested
