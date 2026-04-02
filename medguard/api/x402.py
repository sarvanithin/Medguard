"""
x402 / MPP payment middleware for MedGuard.

Free paths:  /, /health, /v1/health, /docs, /redoc, /openapi.json, /ui/*
Paid paths:  /v1/chat ($0.02), /v1/check/phi ($0.01), /v1/check/drug-interactions ($0.01)

Requests without a valid X-PAYMENT header receive HTTP 402 with pricing info.
Requests with X-PAYMENT: Bearer <token> where token == MPP_SECRET_KEY bypass the gate.
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


def _valid_payment(token: str, path: str) -> bool:
    """Accept Bearer tokens that are HMAC-SHA256(secret, path:minute_window)."""
    minute = str(int(time.time()) // 60)
    for window in (minute, str(int(minute) - 1)):
        expected = hmac.new(
            _SECRET.encode(),
            f"{path}:{window}".encode(),
            hashlib.sha256,
        ).hexdigest()
        if hmac.compare_digest(expected, token):
            return True
    # Also accept raw secret for simple integrations / demos
    if hmac.compare_digest(_SECRET, token):
        return True
    return False


class X402Middleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not _ENABLED:
            return await call_next(request)

        path = request.url.path

        # Always free
        if path in FREE_PATHS or path.startswith("/ui") or path.startswith("/static"):
            return await call_next(request)

        # Only gate POST endpoints that are in the price list
        if request.method != "POST" or path not in ENDPOINT_PRICES:
            return await call_next(request)

        # Check payment header
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
                    "For demo access use X-PAYMENT: Bearer medguard-dev-secret"
                ),
            },
            headers={
                "X-PAYMENT-REQUIRED": "true",
                "X-PRICE-USD": price,
                "X-PROTOCOL": "x402",
            },
        )
