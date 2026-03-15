"""
Guardrail pipeline orchestrator.

Executes the full safety pipeline across 4 stages:
  1. Input stage: PHI redaction + scope enforcement
  2. Pre-LLM stage: drug safety check on input
  3. LLM call
  4. Post-LLM stage: hallucination detection on output

Each stage runs in its own try/except so an API failure in drug safety
never blocks the entire request. Failures are logged and added to warnings.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator

import structlog

if TYPE_CHECKING:
    from medguard.config import MedGuardConfig
    from medguard.guardrails.drug_safety import DrugSafetyResult
    from medguard.guardrails.hallucination import HallucinationResult
    from medguard.guardrails.phi import PHIResult
    from medguard.guardrails.scope import ScopeResult
    from medguard.guardrails.protocols import LLMCallerProtocol

log = structlog.get_logger(__name__)


@dataclass
class PipelineContext:
    """Mutable context passed through all pipeline stages."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_input: str = ""
    processed_input: str = ""       # after PHI redaction
    llm_response: str | None = None
    processed_output: str | None = None  # after hallucination annotation

    # Stage results
    phi_result: "PHIResult | None" = None
    scope_result: "ScopeResult | None" = None
    drug_result: "DrugSafetyResult | None" = None
    hallucination_result: "HallucinationResult | None" = None

    # Control flow
    blocked: bool = False
    block_reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Timing
    started_at: float = field(default_factory=time.monotonic)
    completed_at: float | None = None

    @property
    def processing_time_ms(self) -> float:
        end = self.completed_at or time.monotonic()
        return (end - self.started_at) * 1000

    def final_output(self) -> str:
        """Return the safest available output text."""
        if self.processed_output is not None:
            return self.processed_output
        if self.llm_response is not None:
            return self.llm_response
        return ""


class GuardrailPipeline:
    """
    Executes the medguard safety pipeline.

    Instantiated once per MedGuard instance. All per-request state is
    held in PipelineContext, so the pipeline itself is stateless.
    """

    def __init__(
        self,
        config: "MedGuardConfig",
        phi_detector=None,
        scope_enforcer=None,
        drug_checker=None,
        hallucination_detector=None,
        llm_caller: "LLMCallerProtocol | None" = None,
    ) -> None:
        self.config = config
        self._phi = phi_detector
        self._scope = scope_enforcer
        self._drug = drug_checker
        self._hallucination = hallucination_detector
        self._llm = llm_caller

    async def run(self, user_input: str) -> PipelineContext:
        """Run the full pipeline. Returns context with all results."""
        ctx = PipelineContext(
            original_input=user_input,
            processed_input=user_input,
        )

        await self._run_input_stage(ctx)
        if ctx.blocked:
            ctx.completed_at = time.monotonic()
            return ctx

        await self._run_pre_llm_stage(ctx)
        if ctx.blocked:
            ctx.completed_at = time.monotonic()
            return ctx

        if self._llm is not None:
            try:
                ctx.llm_response = await self._llm.call(ctx.processed_input)
                ctx.processed_output = ctx.llm_response
            except Exception as exc:
                log.error("llm_call_failed", error=str(exc), request_id=ctx.request_id)
                ctx.errors.append(f"LLM call failed: {exc}")
                ctx.completed_at = time.monotonic()
                return ctx

        await self._run_post_llm_stage(ctx)

        ctx.completed_at = time.monotonic()
        log.info(
            "pipeline_complete",
            request_id=ctx.request_id,
            blocked=ctx.blocked,
            warnings=len(ctx.warnings),
            duration_ms=round(ctx.processing_time_ms, 1),
        )
        return ctx

    async def run_streaming(
        self, user_input: str
    ) -> tuple[PipelineContext, "AsyncIterator[str] | None"]:
        """
        Run input/pre-LLM checks synchronously, then return a streaming
        iterator for the LLM response.

        Returns (context, stream_iter). If context.blocked, stream_iter is None.
        Post-LLM guardrails run on the full buffer after streaming completes;
        the caller is responsible for consuming the iterator from _stream_with_postcheck().
        """
        ctx = PipelineContext(
            original_input=user_input,
            processed_input=user_input,
        )

        await self._run_input_stage(ctx)
        if ctx.blocked:
            ctx.completed_at = time.monotonic()
            return ctx, None

        await self._run_pre_llm_stage(ctx)
        if ctx.blocked:
            ctx.completed_at = time.monotonic()
            return ctx, None

        if self._llm is None:
            ctx.completed_at = time.monotonic()
            return ctx, None

        stream = self._stream_with_postcheck(ctx)
        return ctx, stream

    async def _stream_with_postcheck(
        self, ctx: PipelineContext
    ) -> AsyncIterator[str]:
        """Stream LLM tokens, buffer them, run post-LLM checks after completion."""
        buffer: list[str] = []
        try:
            async for token in self._llm.call_stream(ctx.processed_input):
                buffer.append(token)
                yield token
        except Exception as exc:
            log.error("llm_stream_failed", error=str(exc), request_id=ctx.request_id)
            ctx.errors.append(f"LLM stream failed: {exc}")
            return

        ctx.llm_response = "".join(buffer)
        ctx.processed_output = ctx.llm_response

        await self._run_post_llm_stage(ctx)
        ctx.completed_at = time.monotonic()

        # Yield any annotations appended by post-LLM stage
        if ctx.processed_output and ctx.processed_output != ctx.llm_response:
            suffix = ctx.processed_output[len(ctx.llm_response):]
            if suffix:
                yield suffix

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    async def _run_input_stage(self, ctx: PipelineContext) -> None:
        """Stage 1: PHI redaction + clinical scope check."""
        # PHI detection
        if self._phi is not None and self.config.guardrails.phi_detection.enabled:
            try:
                phi_result = self._phi.detect(ctx.processed_input)
                ctx.phi_result = phi_result
                if phi_result.phi_detected:
                    ctx.processed_input = phi_result.processed
                    log.info(
                        "phi_detected",
                        entities=[m.entity_type for m in phi_result.matches],
                        request_id=ctx.request_id,
                    )
                    # Check mode from the detector's own config (may differ from pipeline config)
                    phi_mode = getattr(self._phi, "config", None)
                    phi_mode = getattr(phi_mode, "mode", None) or self.config.guardrails.phi_detection.mode
                    if phi_mode == "block":
                        ctx.blocked = True
                        ctx.block_reason = (
                            "Input contains protected health information (PHI). "
                            "Please remove personal identifiers before submitting."
                        )
                        return
            except Exception as exc:
                log.warning("phi_check_failed", error=str(exc), request_id=ctx.request_id)
                ctx.errors.append(f"PHI check error: {exc}")

        # Scope enforcement
        if self._scope is not None and self.config.guardrails.scope_enforcement.enabled:
            try:
                scope_result = self._scope.check(ctx.processed_input)
                ctx.scope_result = scope_result
                if not scope_result.in_scope:
                    if scope_result.action_taken == "block":
                        ctx.blocked = True
                        ctx.block_reason = scope_result.reason
                        return
                    else:
                        ctx.warnings.append(scope_result.reason or "Out-of-scope query detected.")
            except Exception as exc:
                log.warning("scope_check_failed", error=str(exc), request_id=ctx.request_id)
                ctx.errors.append(f"Scope check error: {exc}")

    async def _run_pre_llm_stage(self, ctx: PipelineContext) -> None:
        """Stage 2: Drug interaction check on the (possibly redacted) input."""
        if self._drug is None or not self.config.guardrails.drug_safety.enabled:
            return
        try:
            drug_result = await self._drug.check(ctx.processed_input)
            ctx.drug_result = drug_result
            if drug_result.blocked:
                ctx.blocked = True
                interaction_summary = "; ".join(
                    f"{i.drug_a} + {i.drug_b} ({i.severity.value})"
                    for i in drug_result.interactions
                )
                ctx.block_reason = (
                    f"Drug safety concern detected: {interaction_summary}. "
                    "Please consult a healthcare professional before proceeding."
                )
            elif drug_result.warnings:
                ctx.warnings.extend(drug_result.warnings)
        except Exception as exc:
            log.warning("drug_check_failed", error=str(exc), request_id=ctx.request_id)
            ctx.errors.append(f"Drug safety check error: {exc}")

    async def _run_post_llm_stage(self, ctx: PipelineContext) -> None:
        """Stage 4: Hallucination detection on LLM output."""
        if (
            self._hallucination is None
            or not self.config.guardrails.hallucination_detection.enabled
            or ctx.llm_response is None
        ):
            return
        try:
            hall_result = await self._hallucination.check(ctx.llm_response)
            ctx.hallucination_result = hall_result
            if hall_result.flags:
                flag_types = list({f.type.value for f in hall_result.flags})
                ctx.warnings.append(
                    f"Medical accuracy notice: {', '.join(flag_types)} detected in response."
                )
                # Use annotated text (with inline [WARNING: ...] markers)
                ctx.processed_output = hall_result.annotated_text
                if hall_result.blocked:
                    ctx.blocked = True
                    ctx.block_reason = (
                        "Response contains potentially inaccurate medical information "
                        f"(hallucination score: {hall_result.hallucination_score:.2f})."
                    )
        except Exception as exc:
            log.warning(
                "hallucination_check_failed", error=str(exc), request_id=ctx.request_id
            )
            ctx.errors.append(f"Hallucination check error: {exc}")
