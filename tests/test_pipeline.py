"""
End-to-end pipeline tests.

Uses mocked LLM and mocked external APIs.
Tests blocking, warning, and streaming behaviors.
"""
import pytest
import respx
import httpx
from unittest.mock import AsyncMock, MagicMock

from medguard.config import MedGuardConfig, GuardrailsConfig, PHIConfig, ScopeConfig
from medguard.guardrails.pipeline import GuardrailPipeline, PipelineContext
from medguard.guardrails.phi import PHIDetector
from medguard.guardrails.scope import ScopeEnforcer


@pytest.fixture
def full_config():
    return MedGuardConfig()


@pytest.fixture
def phi_detector():
    from medguard.config import PHIConfig
    return PHIDetector(PHIConfig(enabled=True, mode="redact", engine="regex"))


@pytest.fixture
def scope_enforcer():
    from medguard.config import ScopeConfig
    return ScopeEnforcer(ScopeConfig(enabled=True, action="warn"))


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.call = AsyncMock(return_value="Metformin is commonly used for type 2 diabetes.")
    return llm


class TestPipelineInputStage:
    @pytest.mark.asyncio
    async def test_phi_redacted_in_processed_input(self, full_config, phi_detector, scope_enforcer, mock_llm):
        pipeline = GuardrailPipeline(
            config=full_config,
            phi_detector=phi_detector,
            scope_enforcer=scope_enforcer,
            drug_checker=None,
            hallucination_detector=None,
            llm_caller=mock_llm,
        )
        ctx = await pipeline.run("Patient SSN 123-45-6789 asks about metformin")
        assert ctx.phi_result is not None
        assert ctx.phi_result.phi_detected
        assert "123-45-6789" not in ctx.processed_input

    @pytest.mark.asyncio
    async def test_phi_block_mode_blocks_pipeline(self, full_config, scope_enforcer, mock_llm):
        phi_detector = PHIDetector(PHIConfig(enabled=True, mode="block", engine="regex"))
        pipeline = GuardrailPipeline(
            config=full_config,
            phi_detector=phi_detector,
            scope_enforcer=scope_enforcer,
            drug_checker=None,
            hallucination_detector=None,
            llm_caller=mock_llm,
        )
        ctx = await pipeline.run("My SSN is 123-45-6789, what drugs should I take?")
        assert ctx.blocked
        assert ctx.block_reason is not None
        # LLM should NOT have been called
        mock_llm.call.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_out_of_scope_adds_warning(self, full_config, phi_detector, scope_enforcer, mock_llm):
        pipeline = GuardrailPipeline(
            config=full_config,
            phi_detector=phi_detector,
            scope_enforcer=scope_enforcer,
            drug_checker=None,
            hallucination_detector=None,
            llm_caller=mock_llm,
        )
        ctx = await pipeline.run("Can I sue my doctor for malpractice?")
        assert not ctx.blocked  # warn mode, not block
        assert len(ctx.warnings) > 0

    @pytest.mark.asyncio
    async def test_out_of_scope_block_mode_blocks(self, full_config, phi_detector, mock_llm):
        scope_enforcer = ScopeEnforcer(ScopeConfig(enabled=True, action="block"))
        pipeline = GuardrailPipeline(
            config=full_config,
            phi_detector=phi_detector,
            scope_enforcer=scope_enforcer,
            drug_checker=None,
            hallucination_detector=None,
            llm_caller=mock_llm,
        )
        ctx = await pipeline.run("Can I sue my doctor for malpractice?")
        assert ctx.blocked
        mock_llm.call.assert_not_awaited()


class TestPipelineLLMCall:
    @pytest.mark.asyncio
    async def test_llm_response_captured(self, full_config, phi_detector, scope_enforcer, mock_llm):
        pipeline = GuardrailPipeline(
            config=full_config,
            phi_detector=phi_detector,
            scope_enforcer=scope_enforcer,
            drug_checker=None,
            hallucination_detector=None,
            llm_caller=mock_llm,
        )
        ctx = await pipeline.run("What are the side effects of metformin?")
        assert ctx.llm_response == "Metformin is commonly used for type 2 diabetes."
        assert ctx.processed_output is not None

    @pytest.mark.asyncio
    async def test_llm_failure_handled_gracefully(self, full_config, phi_detector, scope_enforcer):
        llm = MagicMock()
        llm.call = AsyncMock(side_effect=Exception("API timeout"))
        pipeline = GuardrailPipeline(
            config=full_config,
            phi_detector=phi_detector,
            scope_enforcer=scope_enforcer,
            drug_checker=None,
            hallucination_detector=None,
            llm_caller=llm,
        )
        ctx = await pipeline.run("What is metformin?")
        assert len(ctx.errors) > 0
        assert ctx.llm_response is None

    @pytest.mark.asyncio
    async def test_no_llm_returns_context(self, full_config, phi_detector, scope_enforcer):
        pipeline = GuardrailPipeline(
            config=full_config,
            phi_detector=phi_detector,
            scope_enforcer=scope_enforcer,
            drug_checker=None,
            hallucination_detector=None,
            llm_caller=None,  # no LLM
        )
        ctx = await pipeline.run("What is metformin?")
        assert ctx.llm_response is None
        assert not ctx.blocked


class TestPipelineGuardrailIsolation:
    @pytest.mark.asyncio
    async def test_guardrail_failure_does_not_block_request(
        self, full_config, phi_detector, scope_enforcer, mock_llm
    ):
        """An exception in a guardrail should not block the entire pipeline."""
        drug_checker = MagicMock()
        drug_checker.check = AsyncMock(side_effect=Exception("RxNorm timeout"))

        pipeline = GuardrailPipeline(
            config=full_config,
            phi_detector=phi_detector,
            scope_enforcer=scope_enforcer,
            drug_checker=drug_checker,
            hallucination_detector=None,
            llm_caller=mock_llm,
        )
        ctx = await pipeline.run("Patient takes warfarin and aspirin.")
        # Pipeline should continue despite drug check failure
        assert not ctx.blocked
        assert len(ctx.errors) > 0  # error logged
        # LLM was still called
        mock_llm.call.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_phi_failure_does_not_block(self, full_config, scope_enforcer, mock_llm):
        """PHI engine failure should not block the pipeline."""
        bad_phi = MagicMock()
        bad_phi.detect = MagicMock(side_effect=Exception("PHI engine crashed"))

        pipeline = GuardrailPipeline(
            config=full_config,
            phi_detector=bad_phi,
            scope_enforcer=scope_enforcer,
            drug_checker=None,
            hallucination_detector=None,
            llm_caller=mock_llm,
        )
        ctx = await pipeline.run("What is metformin?")
        assert not ctx.blocked
        assert len(ctx.errors) > 0


class TestPipelineMetadata:
    @pytest.mark.asyncio
    async def test_request_id_set(self, full_config, mock_llm):
        pipeline = GuardrailPipeline(
            config=full_config,
            phi_detector=None,
            scope_enforcer=None,
            drug_checker=None,
            hallucination_detector=None,
            llm_caller=mock_llm,
        )
        ctx = await pipeline.run("Hello")
        assert ctx.request_id
        assert len(ctx.request_id) == 36  # UUID4 format

    @pytest.mark.asyncio
    async def test_processing_time_recorded(self, full_config, mock_llm):
        pipeline = GuardrailPipeline(
            config=full_config,
            phi_detector=None,
            scope_enforcer=None,
            drug_checker=None,
            hallucination_detector=None,
            llm_caller=mock_llm,
        )
        ctx = await pipeline.run("Hello")
        assert ctx.completed_at is not None
        assert ctx.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_final_output_returns_response(self, full_config, mock_llm):
        pipeline = GuardrailPipeline(
            config=full_config,
            phi_detector=None,
            scope_enforcer=None,
            drug_checker=None,
            hallucination_detector=None,
            llm_caller=mock_llm,
        )
        ctx = await pipeline.run("What is metformin?")
        assert ctx.final_output() == "Metformin is commonly used for type 2 diabetes."
