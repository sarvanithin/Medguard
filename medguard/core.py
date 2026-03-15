"""
MedGuard — primary public API.

Usage:

    # Minimal (config from ~/.medguard/config.json)
    mg = MedGuard()

    # Check text for safety issues (no LLM call)
    result = mg.check("Patient SSN 123-45-6789 is taking warfarin and aspirin")
    print(result.blocked, result.warnings)

    # Full guardrailed LLM chat
    response = asyncio.run(mg.achat("What are the side effects of metformin?"))

    # As FastAPI app
    app = mg.create_app()

    # NeMo Guardrails integration
    actions = mg.as_nemo_actions()
"""
from __future__ import annotations

import asyncio
import importlib.metadata
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from medguard.config import MedGuardConfig, get_config
from medguard.guardrails.phi import PHIDetector
from medguard.guardrails.pipeline import GuardrailPipeline, PipelineContext
from medguard.guardrails.scope import ScopeEnforcer

if TYPE_CHECKING:
    from medguard.guardrails.drug_safety import DrugSafetyChecker
    from medguard.guardrails.fact_check import FactVerifier
    from medguard.guardrails.hallucination import HallucinationDetector
    from medguard.guardrails.protocols import LLMCallerProtocol

log = structlog.get_logger(__name__)


class MedGuard:
    """
    Healthcare-specific LLM guardrail middleware.

    Wires all safety components into a pipeline and exposes both sync
    and async interfaces. Thread-safe; the pipeline itself is stateless.
    """

    def __init__(
        self,
        config: MedGuardConfig | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = MedGuardConfig.load(Path(config_path))
        else:
            self.config = get_config()

        self.phi_detector: PHIDetector | None = None
        self.scope_enforcer: ScopeEnforcer | None = None
        self.drug_checker: DrugSafetyChecker | None = None
        self.hallucination_detector: HallucinationDetector | None = None
        self.fact_verifier: FactVerifier | None = None
        self._llm_caller: LLMCallerProtocol | None = None

        self._build_components()
        self.pipeline = self._build_pipeline()

    # ------------------------------------------------------------------
    # Public sync / async API
    # ------------------------------------------------------------------

    def check(self, text: str) -> PipelineContext:
        """
        Synchronous guardrail check (no LLM call).

        Runs PHI detection, scope enforcement, and drug safety check on
        the input text. Returns a PipelineContext with all results.
        """
        return asyncio.run(self.acheck(text))

    async def acheck(self, text: str) -> PipelineContext:
        """Async guardrail check (no LLM call)."""
        pipeline = GuardrailPipeline(
            config=self.config,
            phi_detector=self.phi_detector,
            scope_enforcer=self.scope_enforcer,
            drug_checker=self.drug_checker,
            hallucination_detector=self.hallucination_detector,
            llm_caller=None,  # no LLM call
        )
        return await pipeline.run(text)

    async def achat(self, message: str) -> str:
        """Full pipeline: guardrails + LLM call + output guardrails."""
        ctx = await self.pipeline.run(message)
        if ctx.blocked:
            return f"[BLOCKED] {ctx.block_reason}"
        output = ctx.final_output()
        if ctx.warnings:
            warning_block = "\n\n---\n**Safety notices:** " + " | ".join(ctx.warnings)
            output += warning_block
        return output

    async def achat_stream(self, message: str) -> AsyncIterator[str]:
        """Streaming version of achat. Yields tokens as they arrive."""
        ctx, stream_iter = await self.pipeline.run_streaming(message)
        if ctx.blocked:
            yield f"[BLOCKED] {ctx.block_reason}"
            return
        if stream_iter is None:
            return
        async for token in stream_iter:
            yield token

    def create_app(self):
        """Create a FastAPI app pre-configured with this MedGuard instance."""
        from medguard.api.app import create_app
        return create_app(medguard_instance=self)

    def as_nemo_actions(self) -> dict:
        """Return NeMo Guardrails-compatible action handlers."""
        from medguard.integrations.nemo import build_nemo_actions
        return build_nemo_actions(self)

    # ------------------------------------------------------------------
    # Component construction
    # ------------------------------------------------------------------

    def _build_components(self) -> None:
        cfg = self.config

        # PHI detector
        if cfg.guardrails.phi_detection.enabled:
            phi_engine = cfg.guardrails.phi_detection.engine
            # Check for registered entry-point engines
            registered_engines = _discover_entry_points("medguard.phi_engines")
            if phi_engine in registered_engines:
                engine_cls = registered_engines[phi_engine]
                try:
                    self.phi_detector = PHIDetector.__new__(PHIDetector)
                    self.phi_detector.config = cfg.guardrails.phi_detection
                    self.phi_detector._engine = engine_cls()
                except Exception as exc:
                    log.warning(
                        "custom_phi_engine_failed",
                        engine=phi_engine,
                        error=str(exc),
                    )
                    self.phi_detector = PHIDetector(cfg.guardrails.phi_detection)
            else:
                self.phi_detector = PHIDetector(cfg.guardrails.phi_detection)

        # Scope enforcer
        if cfg.guardrails.scope_enforcement.enabled:
            self.scope_enforcer = ScopeEnforcer(cfg.guardrails.scope_enforcement)

        # Drug safety checker (requires async setup, done lazily)
        if cfg.guardrails.drug_safety.enabled:
            try:
                import httpx

                from medguard.guardrails.drug_safety import DrugSafetyChecker
                from medguard.knowledge.openfda import OpenFDAClient
                from medguard.knowledge.rxnorm import RxNormClient

                http_client = httpx.AsyncClient(timeout=cfg.guardrails.drug_safety.api_timeout_seconds)
                rxnorm = RxNormClient(cfg.guardrails.drug_safety, http_client)
                openfda = OpenFDAClient(cfg.guardrails.drug_safety, http_client)
                self.drug_checker = DrugSafetyChecker(
                    cfg.guardrails.drug_safety, rxnorm, openfda
                )
            except ImportError:
                log.warning("drug_safety_deps_missing")

        # Hallucination detector
        if cfg.guardrails.hallucination_detection.enabled:
            try:
                import httpx

                from medguard.guardrails.hallucination import HallucinationDetector
                from medguard.knowledge.rxnorm import RxNormClient
                from medguard.knowledge.snomed import SNOMEDClient

                if not hasattr(self, '_http_client'):
                    self._http_client = httpx.AsyncClient(timeout=5.0)
                rxnorm = RxNormClient(cfg.guardrails.drug_safety, self._http_client)
                snomed = SNOMEDClient(self._http_client)
                self.hallucination_detector = HallucinationDetector(
                    cfg.guardrails.hallucination_detection, rxnorm, snomed
                )
            except ImportError:
                log.warning("hallucination_deps_missing")

        # PubMed fact verifier (opt-in)
        if cfg.guardrails.fact_checking.enabled:
            try:
                import httpx

                from medguard.guardrails.fact_check import FactVerifier
                from medguard.knowledge.pubmed import PubMedClient

                if not hasattr(self, "_http_client"):
                    self._http_client = httpx.AsyncClient(timeout=10.0)
                pubmed = PubMedClient(
                    self._http_client,
                    max_results=cfg.guardrails.fact_checking.max_claims_per_response,
                )
                self.fact_verifier = FactVerifier(
                    pubmed,
                    confidence_threshold=cfg.guardrails.fact_checking.confidence_threshold,
                )
            except Exception as exc:
                log.warning("fact_verifier_init_failed", error=str(exc))

        # LLM caller
        try:
            self._llm_caller = _build_llm_caller(cfg)
        except Exception as exc:
            log.warning("llm_caller_init_failed", error=str(exc))

    def _build_pipeline(self) -> GuardrailPipeline:
        return GuardrailPipeline(
            config=self.config,
            phi_detector=self.phi_detector,
            scope_enforcer=self.scope_enforcer,
            drug_checker=self.drug_checker,
            hallucination_detector=self.hallucination_detector,
            fact_verifier=self.fact_verifier,
            llm_caller=self._llm_caller,
        )


def _build_llm_caller(config: MedGuardConfig) -> LLMCallerProtocol | None:
    provider = config.llm.provider
    if provider == "anthropic":
        from medguard.integrations.anthropic import AnthropicCaller
        return AnthropicCaller(config.llm)
    elif provider == "openai":
        from medguard.integrations.openai import OpenAICaller
        return OpenAICaller(config.llm)
    elif provider == "ollama":
        from medguard.config import LLMConfig
        from medguard.integrations.openai import OpenAICaller
        ollama_config = LLMConfig(
            provider="ollama",
            model=config.llm.model,
            base_url=config.llm.base_url or "http://localhost:11434/v1",
            api_key_env=config.llm.api_key_env,
            timeout_seconds=config.llm.timeout_seconds,
            max_tokens=config.llm.max_tokens,
            system_prompt=config.llm.system_prompt,
        )
        return OpenAICaller(ollama_config)
    return None


def _discover_entry_points(group: str) -> dict[str, type]:
    """Load registered extension plugins from pyproject.toml entry points."""
    engines: dict[str, type] = {}
    try:
        for ep in importlib.metadata.entry_points(group=group):
            try:
                engines[ep.name] = ep.load()
            except Exception as exc:
                log.warning("entry_point_load_failed", name=ep.name, error=str(exc))
    except Exception:
        pass
    return engines
