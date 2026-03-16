"""
FastAPI route handlers for medguard.

Routes:
    POST /v1/chat                  — guardrailed LLM chat
    POST /v1/check/phi             — standalone PHI check
    POST /v1/check/drug-interactions — standalone drug interaction check
    GET  /v1/health                — dependency health check
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import TYPE_CHECKING

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from medguard.api.models import (
    ChatRequest,
    ChatResponse,
    ContraindicationInfo,
    DependencyStatus,
    DrugInteractionRequest,
    DrugInteractionResponse,
    GuardrailAnnotation,
    HealthResponse,
    InteractionInfo,
    PHICheckRequest,
    PHICheckResponse,
    PHIMatchInfo,
)

if TYPE_CHECKING:
    from medguard.core import MedGuard

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/v1")

_start_time = time.monotonic()


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------

_medguard_instance: MedGuard | None = None


def set_medguard_instance(instance: MedGuard) -> None:
    global _medguard_instance
    _medguard_instance = instance


def get_medguard() -> MedGuard:
    if _medguard_instance is None:
        from medguard.core import MedGuard
        set_medguard_instance(MedGuard())
    return _medguard_instance


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    mg: MedGuard = Depends(get_medguard),
) -> ChatResponse | StreamingResponse:
    """
    Main guardrailed chat endpoint.

    Runs the full medguard pipeline:
    1. PHI detection/redaction
    2. Clinical scope check
    3. Drug interaction check (on input)
    4. LLM call
    5. Hallucination detection (on output)

    For stream=True, returns Server-Sent Events.
    Each SSE chunk: data: {"delta": "...", "done": false}
    Final SSE chunk: data: {"delta": "", "done": true, "guardrails": [...], "warnings": [...]}
    """
    user_text = request.messages[-1].content if request.messages else ""

    if request.stream:
        ctx, stream_iter = await mg.pipeline.run_streaming(user_text)
        if ctx.blocked:
            raise HTTPException(status_code=422, detail=ctx.block_reason)

        async def event_generator() -> AsyncIterator[str]:
            if stream_iter is None:
                yield _sse_done(ctx)
                return
            async for token in stream_iter:
                yield f"data: {_sse_delta(token)}\n\n"
            yield f"data: {_sse_done(ctx)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    ctx = await mg.pipeline.run(user_text)

    guardrails = _build_guardrail_annotations(ctx)
    response = ChatResponse(
        id=ctx.request_id,
        content=ctx.block_reason if ctx.blocked else (ctx.final_output() or ""),
        guardrails=guardrails,
        blocked=ctx.blocked,
        block_reason=ctx.block_reason,
        phi_redacted=bool(ctx.phi_result and ctx.phi_result.phi_detected),
        warnings=ctx.warnings,
        processing_time_ms=round(ctx.processing_time_ms, 1),
    )
    # Attach full trace as extra field for the UI
    response_dict = response.model_dump()
    response_dict["trace"] = _build_trace(ctx)
    from fastapi.responses import JSONResponse
    return JSONResponse(content=response_dict)


@router.post("/check/phi", response_model=PHICheckResponse)
async def check_phi(
    request: PHICheckRequest,
    mg: MedGuard = Depends(get_medguard),
) -> PHICheckResponse:
    """Standalone PHI detection endpoint."""
    if mg.phi_detector is None:
        raise HTTPException(status_code=503, detail="PHI detection is not enabled.")

    result = mg.phi_detector.detect(request.text)
    return PHICheckResponse(
        phi_detected=result.phi_detected,
        matches=[
            PHIMatchInfo(
                entity_type=m.entity_type,
                start=m.start,
                end=m.end,
                text=m.text if request.mode == "detect" else "[REDACTED]",
                confidence=m.confidence,
            )
            for m in result.matches
        ],
        redacted_text=result.processed if request.mode == "redact" else None,
        engine_used=result.engine_used,
    )


@router.post("/check/drug-interactions", response_model=DrugInteractionResponse)
async def check_drug_interactions(
    request: DrugInteractionRequest,
    mg: MedGuard = Depends(get_medguard),
) -> DrugInteractionResponse:
    """Standalone drug interaction and contraindication check."""
    if mg.drug_checker is None:
        raise HTTPException(status_code=503, detail="Drug safety checking is not enabled.")

    # Build a pseudo-input text listing the drugs
    drug_text = "Patient is taking " + " and ".join(request.drugs) + "."
    result = await mg.drug_checker.check(drug_text)

    interactions = [
        InteractionInfo(
            drug_a=i.drug_a,
            drug_b=i.drug_b,
            severity=i.severity.value,
            description=i.description,
            source=i.source,
        )
        for i in result.interactions
    ]

    contraindications: list[ContraindicationInfo] = []
    if request.include_contraindications and result.contraindications:
        contraindications = [
            ContraindicationInfo(
                drug=c.drug,
                condition=c.condition,
                description=c.description,
            )
            for c in result.contraindications
        ]

    drugs_not_found = [
        d.raw_name for d in result.drugs_found if d.canonical_name is None
    ]

    return DrugInteractionResponse(
        interactions=interactions,
        contraindications=contraindications,
        highest_severity=result.highest_severity.value,
        drugs_not_found=drugs_not_found,
    )


@router.get("/health", response_model=HealthResponse)
async def health_check(
    mg: MedGuard = Depends(get_medguard),
) -> HealthResponse:
    """
    Health check endpoint. Pings external APIs and checks component availability.
    Returns 'degraded' (not 'unhealthy') if only one API is unavailable.
    """
    rxnorm_ok, openfda_ok = await asyncio.gather(
        _ping_rxnorm(),
        _ping_openfda(),
        return_exceptions=False,
    )

    phi_engine = (
        type(mg.phi_detector._engine).__name__ if mg.phi_detector else "disabled"
    )

    dep_status = DependencyStatus(
        rxnorm_api=rxnorm_ok,
        openfda_api=openfda_ok,
        phi_engine=phi_engine,
        cache_available=True,  # diskcache is always local
    )

    api_count = sum([rxnorm_ok, openfda_ok])
    if api_count == 2:
        status = "healthy"
    elif api_count == 1:
        status = "degraded"
    else:
        status = "unhealthy"

    cfg = mg.config.guardrails
    return HealthResponse(
        status=status,
        dependencies=dep_status,
        version=_get_version(),
        uptime_seconds=round(time.monotonic() - _start_time, 1),
        guardrails_enabled={
            "phi_detection": cfg.phi_detection.enabled,
            "drug_safety": cfg.drug_safety.enabled,
            "scope_enforcement": cfg.scope_enforcement.enabled,
            "hallucination_detection": cfg.hallucination_detection.enabled,
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_guardrail_annotations(ctx) -> list[GuardrailAnnotation]:
    annotations = []

    if ctx.phi_result is not None:
        annotations.append(
            GuardrailAnnotation(
                guardrail="phi_detection",
                triggered=ctx.phi_result.phi_detected,
                details=(
                    f"{len(ctx.phi_result.matches)} entity(ies) detected"
                    if ctx.phi_result.phi_detected
                    else None
                ),
            )
        )

    if ctx.scope_result is not None:
        annotations.append(
            GuardrailAnnotation(
                guardrail="scope_enforcement",
                triggered=not ctx.scope_result.in_scope,
                details=ctx.scope_result.category.value,
            )
        )

    if ctx.drug_result is not None:
        annotations.append(
            GuardrailAnnotation(
                guardrail="drug_safety",
                triggered=bool(ctx.drug_result.interactions),
                severity=ctx.drug_result.highest_severity.value,
                details=(
                    f"{len(ctx.drug_result.interactions)} interaction(s) found"
                    if ctx.drug_result.interactions
                    else None
                ),
            )
        )

    if ctx.hallucination_result is not None:
        annotations.append(
            GuardrailAnnotation(
                guardrail="hallucination_detection",
                triggered=bool(ctx.hallucination_result.flags),
                severity=f"{ctx.hallucination_result.hallucination_score:.2f}",
                details=(
                    f"{len(ctx.hallucination_result.flags)} flag(s)"
                    if ctx.hallucination_result.flags
                    else None
                ),
            )
        )

    return annotations


def _sse_delta(token: str) -> str:
    import json
    return json.dumps({"delta": token, "done": False})


def _sse_done(ctx) -> str:
    import json
    annotations = _build_guardrail_annotations(ctx)
    return json.dumps({
        "delta": "",
        "done": True,
        "blocked": ctx.blocked,
        "block_reason": ctx.block_reason,
        "guardrails": [a.model_dump() for a in annotations],
        "warnings": ctx.warnings,
        "trace": _build_trace(ctx),
    })


def _build_trace(ctx) -> dict:
    """Build a full pipeline trace for the UI inspector."""
    trace = {
        "stages": [],
        "original_input": ctx.original_input,
        "processed_input": ctx.processed_input,
        "llm_response": ctx.llm_response,
        "final_output": ctx.final_output(),
        "errors": ctx.errors,
    }

    # Stage 1a: PHI
    if ctx.phi_result is not None:
        phi = ctx.phi_result
        trace["stages"].append({
            "stage": "1a",
            "name": "PHI Detection",
            "triggered": phi.phi_detected,
            "engine": phi.engine_used,
            "entities": [
                {"type": m.entity_type, "text": m.text, "confidence": round(m.confidence, 2)}
                for m in phi.matches
            ],
            "redacted_input": phi.processed if phi.phi_detected else None,
        })

    # Stage 1b: Scope
    if ctx.scope_result is not None:
        s = ctx.scope_result
        trace["stages"].append({
            "stage": "1b",
            "name": "Scope Enforcement",
            "triggered": not s.in_scope,
            "category": s.category.value,
            "action": s.action_taken,
            "reason": s.reason,
        })

    # Stage 2: Drug Safety
    if ctx.drug_result is not None:
        dr = ctx.drug_result
        trace["stages"].append({
            "stage": "2",
            "name": "Drug Safety",
            "triggered": bool(dr.interactions),
            "drugs_found": [d.raw_name for d in dr.drugs_found],
            "highest_severity": dr.highest_severity.value,
            "interactions": [
                {
                    "drug_a": i.drug_a, "drug_b": i.drug_b,
                    "severity": i.severity.value, "description": i.description,
                    "source": i.source,
                }
                for i in dr.interactions
            ],
        })

    # Stage 4a: Hallucination
    if ctx.hallucination_result is not None:
        hr = ctx.hallucination_result
        trace["stages"].append({
            "stage": "4a",
            "name": "Hallucination Detection",
            "triggered": bool(hr.flags),
            "score": round(hr.hallucination_score, 3),
            "blocked": hr.blocked,
            "flags": [
                {
                    "type": f.type.value, "text": f.text,
                    "confidence": round(f.confidence, 2), "explanation": f.explanation,
                }
                for f in hr.flags
            ],
            "annotated_text": hr.annotated_text if hr.flags else None,
        })

    return trace


async def _ping_rxnorm() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(
                "https://rxnav.nlm.nih.gov/REST/rxcui.json",
                params={"name": "aspirin"},
            )
            return r.status_code == 200
    except Exception:
        return False


async def _ping_openfda() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(
                "https://api.fda.gov/drug/label.json",
                params={"search": "openfda.generic_name:aspirin", "limit": "1"},
            )
            return r.status_code == 200
    except Exception:
        return False


@lru_cache(maxsize=1)
def _get_version() -> str:
    try:
        from importlib.metadata import version
        return version("medguard")
    except Exception:
        return "0.1.0"
