"""
FastAPI route tests.

Uses httpx.AsyncClient with the app directly — no real network calls for guardrails.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from medguard.api.app import create_app
from medguard.guardrails.phi import PHIMatch, PHIResult
from medguard.guardrails.pipeline import PipelineContext


def _make_mock_medguard(blocked=False, phi_detected=False, warnings=None):
    """Build a minimal MedGuard mock for route testing."""
    mg = MagicMock()
    mg.config = MagicMock()
    mg.config.guardrails.phi_detection.enabled = True
    mg.config.guardrails.drug_safety.enabled = True
    mg.config.guardrails.scope_enforcement.enabled = True
    mg.config.guardrails.hallucination_detection.enabled = True

    ctx = PipelineContext(
        original_input="test",
        processed_input="test",
    )
    ctx.blocked = blocked
    ctx.block_reason = "Blocked for test." if blocked else None
    ctx.warnings = warnings or []
    ctx.llm_response = "This is a safe medical response." if not blocked else None
    ctx.processed_output = ctx.llm_response

    if phi_detected:
        ctx.phi_result = PHIResult(
            original="Patient SSN: 123-45-6789",
            processed="Patient SSN: [REDACTED]",
            matches=[
                PHIMatch(
                    entity_type="SSN",
                    start=13,
                    end=24,
                    text="123-45-6789",
                    confidence=0.95,
                    redacted_text="[REDACTED]",
                )
            ],
            phi_detected=True,
            engine_used="RegexPHIEngine",
        )

    mg.pipeline = MagicMock()
    mg.pipeline.run = AsyncMock(return_value=ctx)
    mg.pipeline.run_streaming = AsyncMock(return_value=(ctx, None))

    # PHI detector
    if phi_detected:
        mg.phi_detector = MagicMock()
        mg.phi_detector._engine = MagicMock()
        mg.phi_detector._engine.__class__.__name__ = "RegexPHIEngine"
        mg.phi_detector.detect = MagicMock(return_value=ctx.phi_result)
    else:
        mg.phi_detector = MagicMock()
        mg.phi_detector._engine.__class__.__name__ = "RegexPHIEngine"
        empty_phi = PHIResult(
            original="test", processed="test", matches=[], phi_detected=False, engine_used="RegexPHIEngine"
        )
        mg.phi_detector.detect = MagicMock(return_value=empty_phi)

    mg.drug_checker = None

    return mg


@pytest.fixture
def client():
    mg = _make_mock_medguard()
    app = create_app(medguard_instance=mg)
    return TestClient(app), mg


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        c, mg = client
        # health endpoint pings real APIs — mock the pipeline dependency check
        response = c.get("/v1/health")
        # May return degraded if APIs unreachable in test env, but should not 500
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "guardrails_enabled" in data
        assert data["guardrails_enabled"]["phi_detection"] is True

    def test_root_redirects_to_ui(self, client):
        c, _ = client
        response = c.get("/", follow_redirects=False)
        assert response.status_code in (301, 302, 307, 308)
        assert "/ui" in response.headers.get("location", "")


class TestChatEndpoint:
    def test_chat_returns_response(self, client):
        c, _ = client
        response = c.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "What are side effects of metformin?"}]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert "guardrails" in data
        assert "blocked" in data
        assert isinstance(data["guardrails"], list)

    def test_chat_blocked_request(self):
        mg = _make_mock_medguard(blocked=True)
        app = create_app(medguard_instance=mg)
        c = TestClient(app)
        response = c.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "test"}]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["blocked"] is True
        assert data["block_reason"] is not None

    def test_chat_with_warnings(self):
        mg = _make_mock_medguard(warnings=["Drug interaction warning: warfarin + aspirin (high)"])
        app = create_app(medguard_instance=mg)
        c = TestClient(app)
        response = c.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "I take warfarin and aspirin"}]},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["warnings"]) > 0

    def test_chat_requires_messages(self, client):
        c, _ = client
        response = c.post("/v1/chat", json={})
        assert response.status_code == 422  # Pydantic validation error

    def test_chat_phi_redacted_flag(self):
        mg = _make_mock_medguard(phi_detected=True)
        app = create_app(medguard_instance=mg)
        c = TestClient(app)
        response = c.post(
            "/v1/chat",
            json={"messages": [{"role": "user", "content": "My SSN is 123-45-6789"}]},
        )
        assert response.status_code == 200


class TestPHIEndpoint:
    def test_phi_detect_mode(self, client):
        c, mg = client
        response = c.post(
            "/v1/check/phi",
            json={"text": "Patient SSN: 123-45-6789", "mode": "detect"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "phi_detected" in data
        assert "matches" in data
        assert "engine_used" in data

    def test_phi_redact_mode_returns_redacted_text(self, client):
        c, mg = client
        phi_result = PHIResult(
            original="SSN: 123-45-6789",
            processed="SSN: [REDACTED]",
            matches=[
                PHIMatch(
                    entity_type="SSN", start=5, end=16,
                    text="123-45-6789", confidence=0.95, redacted_text="[REDACTED]"
                )
            ],
            phi_detected=True,
            engine_used="RegexPHIEngine",
        )
        mg.phi_detector.detect = MagicMock(return_value=phi_result)

        response = c.post(
            "/v1/check/phi",
            json={"text": "SSN: 123-45-6789", "mode": "redact"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["redacted_text"] is not None

    def test_phi_disabled_returns_503(self):
        mg = _make_mock_medguard()
        mg.phi_detector = None
        app = create_app(medguard_instance=mg)
        c = TestClient(app)
        response = c.post("/v1/check/phi", json={"text": "test"})
        assert response.status_code == 503


class TestDrugInteractionEndpoint:
    def test_drug_check_disabled_returns_503(self, client):
        c, _ = client
        # drug_checker is None in the default mock
        response = c.post(
            "/v1/check/drug-interactions",
            json={"drugs": ["warfarin", "aspirin"]},
        )
        assert response.status_code == 503

    def test_drug_check_requires_two_drugs(self, client):
        c, _ = client
        response = c.post(
            "/v1/check/drug-interactions",
            json={"drugs": ["warfarin"]},  # only one drug
        )
        assert response.status_code == 422


class TestAPIModels:
    """Validate Pydantic model schemas."""

    def test_chat_request_valid(self):
        from medguard.api.models import ChatRequest, Message
        req = ChatRequest(messages=[Message(role="user", content="hello")])
        assert req.stream is False
        assert len(req.messages) == 1

    def test_chat_request_stream_flag(self):
        from medguard.api.models import ChatRequest, Message
        req = ChatRequest(
            messages=[Message(role="user", content="hello")],
            stream=True,
        )
        assert req.stream is True

    def test_phi_check_request_defaults(self):
        from medguard.api.models import PHICheckRequest
        req = PHICheckRequest(text="test input")
        assert req.mode == "detect"

    def test_drug_interaction_requires_min_two_drugs(self):
        import pytest

        from medguard.api.models import DrugInteractionRequest
        with pytest.raises(Exception):
            DrugInteractionRequest(drugs=["warfarin"])  # min_length=2

    def test_drug_interaction_valid(self):
        from medguard.api.models import DrugInteractionRequest
        req = DrugInteractionRequest(drugs=["warfarin", "aspirin"])
        assert len(req.drugs) == 2
        assert req.include_contraindications is True
