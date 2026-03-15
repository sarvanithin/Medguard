"""
Pydantic v2 request/response models for the medguard FastAPI layer.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    stream: bool = False
    guardrails_override: dict | None = Field(
        default=None,
        description="Per-request config overrides for guardrail settings.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "messages": [
                    {"role": "user", "content": "What are the side effects of metformin?"}
                ],
                "stream": False,
            }
        }
    )


class GuardrailAnnotation(BaseModel):
    guardrail: str
    triggered: bool
    severity: str | None = None
    details: str | None = None


class ChatResponse(BaseModel):
    id: str
    content: str
    guardrails: list[GuardrailAnnotation]
    blocked: bool
    block_reason: str | None = None
    phi_redacted: bool
    warnings: list[str]
    processing_time_ms: float


class PHICheckRequest(BaseModel):
    text: str
    mode: Literal["detect", "redact"] = "detect"

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Patient John Smith, SSN: 123-45-6789, DOB: 01/15/1980",
                "mode": "redact",
            }
        }
    )


class PHIMatchInfo(BaseModel):
    entity_type: str
    start: int
    end: int
    text: str
    confidence: float


class PHICheckResponse(BaseModel):
    phi_detected: bool
    matches: list[PHIMatchInfo]
    redacted_text: str | None = None
    engine_used: str


class DrugInteractionRequest(BaseModel):
    drugs: list[str] = Field(min_length=2, description="List of drug names to check.")
    include_contraindications: bool = True

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "drugs": ["warfarin", "aspirin"],
                "include_contraindications": True,
            }
        }
    )


class InteractionInfo(BaseModel):
    drug_a: str
    drug_b: str
    severity: str
    description: str
    source: str


class ContraindicationInfo(BaseModel):
    drug: str
    condition: str
    description: str


class DrugInteractionResponse(BaseModel):
    interactions: list[InteractionInfo]
    contraindications: list[ContraindicationInfo]
    highest_severity: str
    drugs_not_found: list[str]


class DependencyStatus(BaseModel):
    rxnorm_api: bool
    openfda_api: bool
    phi_engine: str
    cache_available: bool


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    dependencies: DependencyStatus
    version: str
    uptime_seconds: float
    guardrails_enabled: dict[str, bool]
