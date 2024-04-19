from __future__ import annotations

import asyncio

AutoHTTPProtocol: type[asyncio.Protocol]
try:
    from cog._vendor import httptools  # noqa
except ImportError:  # pragma: no cover
    from cog._vendor.uvicorn.protocols.http.h11_impl import H11Protocol

    AutoHTTPProtocol = H11Protocol
else:  # pragma: no cover
    from cog._vendor.uvicorn.protocols.http.httptools_impl import HttpToolsProtocol

    AutoHTTPProtocol = HttpToolsProtocol
