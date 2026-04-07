"""
x402 / MPP payment middleware for MedGuard.

Always free:    /, /health, /v1/health, /docs, /redoc, /openapi.json, /ui/*
Browser free:   any request with a browser User-Agent or onrender.com origin
                (so the hosted UI never hits the payment gate)
Paid (API):     /v1/chat ($0.02), /v1/check/phi ($0.01), /v1/check/drug-interactions ($0.01)

Agent/API callers must send:  X-PAYMENT: Bearer <token>
  where token = HMAC-SHA256(MPP_SECRET_KEY, "<path>:<minute>")
  or the raw MPP_SECRET_KEY value for simple demo access.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

ENDPOINT_PRICES: dict[str, str] = {
    "/v1/chat": "0.02",
    "/v1/check/phi": "0.01",
    "/v1/check/drug-interactions": "0.01",
}

FREE_PATHS: set[str] = {
    "/",
    "/health",
    "/v1/health",
    "/docs",
    "/redoc",
    "/openapi.json",
}

_SECRET = os.getenv("MPP_SECRET_KEY", "medguard-dev-secret")
_ENABLED = os.getenv("X402_ENABLED", "true").lower() != "false"


def _is_browser(request: Request) -> bool:
    """True for browser/UI requests — bypass payment so the hosted demo works."""
    if "Mozilla/" in request.headers.get("user-agent", ""):
        return True
    if request.headers.get("X-HU-App") == "1":
        return True
    origin = request.headers.get("origin", "")
    referer = request.headers.get("referer", "")
    for header in (origin, referer):
        if header and any(h in header for h in ("onrender.com", "github.io", "localhost", "healthuniverse.com")):
            return True
    return False


def _valid_payment(token: str, path: str) -> bool:
    """Accept HMAC-SHA256(secret, path:minute_window) or raw secret for demos."""
    minute = str(int(time.time()) // 60)
    for window in (minute, str(int(minute) - 1)):
        expected = hmac.new(
            _SECRET.encode(),
            f"{path}:{window}".encode(),
            hashlib.sha256,
        ).hexdigest()
        if hmac.compare_digest(expected, token):
            return True
    if hmac.compare_digest(_SECRET, token):
        return True
    return False


class X402Middleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not _ENABLED:
            return await call_next(request)

        path = request.url.path

        # Always-free paths and static assets
        if path in FREE_PATHS or path.startswith("/ui") or path.startswith("/static"):
            return await call_next(request)

        # Only gate POST endpoints in the price list
        if request.method != "POST" or path not in ENDPOINT_PRICES:
            return await call_next(request)

        # Browser / UI requests bypass payment
        if _is_browser(request):
            return await call_next(request)

        # Validate payment token
        auth = request.headers.get("X-PAYMENT") or request.headers.get("Authorization", "")
        token = auth.removeprefix("Bearer ").strip()

        if token and _valid_payment(token, path):
            response = await call_next(request)
            response.headers["X-PAYMENT-ACCEPTED"] = "true"
            return response

        price = ENDPOINT_PRICES[path]
        return JSONResponse(
            status_code=402,
            content={
                "error": "Payment required",
                "path": path,
                "price_usd": price,
                "protocol": "x402",
                "instructions": (
                    f"Send X-PAYMENT: Bearer <token> where token = "
                    f"HMAC-SHA256(MPP_SECRET_KEY, '{path}:<minute>'). "
                    "Demo token: X-PAYMENT: Bearer medguard-dev-secret"
                ),
            },
            headers={
                "X-PAYMENT-REQUIRED": "true",
                "X-PRICE-USD": price,
                "X-PROTOCOL": "x402",
            },
        )
