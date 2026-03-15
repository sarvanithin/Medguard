"""
FastAPI application factory for medguard.
"""
from __future__ import annotations

import logging

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from medguard.api.routes import router

_DESCRIPTION = """
**medguard** — Healthcare-specific LLM guardrails middleware.

Wraps any LLM with clinical safety layers:
- PHI (Protected Health Information) detection and redaction
- Drug interaction checking and contraindication detection
- Clinical scope enforcement
- Medical hallucination flagging
"""


def create_app(medguard_instance=None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        medguard_instance: Optional pre-built MedGuard instance.
                           If None, one is created lazily on first request.
    """
    from medguard.config import get_config
    config = get_config()

    # Configure structlog
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, config.api.log_level.upper(), logging.INFO)
        ),
    )

    app = FastAPI(
        title="medguard",
        description=_DESCRIPTION,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    if medguard_instance is not None:
        from medguard.api.routes import set_medguard_instance
        set_medguard_instance(medguard_instance)

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "medguard",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/v1/health",
        }

    return app
