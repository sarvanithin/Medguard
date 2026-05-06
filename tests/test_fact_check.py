import json

import pytest

from medguard.config import (
    DrugSafetyConfig,
    FactCheckConfig,
    GuardrailsConfig,
    HallucinationConfig,
    MedGuardConfig,
    PHIConfig,
    ScopeConfig,
)
from medguard.core import MedGuard
from medguard.guardrails.fact_check import AgentFactVerifier
from medguard.knowledge.pubmed import FactEvidence, PubMedArticle


class FakePubMed:
    def __init__(self):
        self.fallback_called = False

    async def search(self, query: str) -> list[str]:
        assert "metformin" in query.lower()
        return ["123", "456"]

    async def fetch_summaries(self, pmids: list[str]) -> list[PubMedArticle]:
        return [
            PubMedArticle(
                pmid="123",
                title="Metformin for type 2 diabetes",
                abstract="",
                journal="Example Journal",
                year="2024",
            ),
            PubMedArticle(
                pmid="456",
                title="Unrelated diabetes review",
                abstract="",
                journal="Example Journal",
                year="2023",
            ),
        ]

    async def fetch_abstracts(self, pmids: list[str]) -> list[PubMedArticle]:
        return [
            PubMedArticle(
                pmid="123",
                title="Metformin for type 2 diabetes",
                abstract="Metformin improved glycemic control in adults with type 2 diabetes.",
            )
        ]

    async def verify_claim(self, claim: str) -> FactEvidence:
        self.fallback_called = True
        return FactEvidence(
            claim=claim,
            total_results=1,
            verified=True,
            confidence=0.6,
            summary="keyword fallback",
        )


class FakeLLM:
    def __init__(self, payload: dict):
        self.payload = payload
        self.prompt = ""

    async def call(self, prompt: str) -> str:
        self.prompt = prompt
        return json.dumps(self.payload)


@pytest.mark.asyncio
async def test_agent_fact_verifier_returns_reasoned_evidence():
    llm = FakeLLM({
        "verdict": "supported",
        "confidence": 0.82,
        "citations": ["123"],
        "reasoning": "The retrieved abstract directly discusses metformin use.",
    })
    verifier = AgentFactVerifier(FakePubMed(), llm, confidence_threshold=0.4)

    evidence = await verifier.verify_claim("Metformin is effective for type 2 diabetes")

    assert evidence.verified is True
    assert evidence.confidence == 0.82
    assert evidence.reasoning == "The retrieved abstract directly discusses metformin use."
    assert [article.pmid for article in evidence.supporting] == ["123"]
    assert "Return only JSON" in llm.prompt
    assert "PMID: 123" in llm.prompt


@pytest.mark.asyncio
async def test_agent_fact_verifier_falls_back_without_llm():
    pubmed = FakePubMed()
    verifier = AgentFactVerifier(pubmed, llm_caller=None, confidence_threshold=0.4)

    evidence = await verifier.verify_claim("Metformin is effective for type 2 diabetes")

    assert pubmed.fallback_called is True
    assert evidence.summary == "keyword fallback"


@pytest.mark.asyncio
async def test_agent_fact_verifier_verify_includes_reasoning_summary():
    llm = FakeLLM({
        "verdict": "inconclusive",
        "confidence": 0.2,
        "citations": ["123"],
        "reasoning": "The abstract discusses the topic but does not support the claim.",
    })
    verifier = AgentFactVerifier(FakePubMed(), llm, confidence_threshold=0.4)

    result = await verifier.verify("Metformin is effective for type 2 diabetes")

    assert result.claims_checked == 1
    assert result.flagged is True
    assert result.pubmed_evidence[0]["reasoning"] == (
        "The abstract discusses the topic but does not support the claim."
    )


def test_medguard_builds_agent_fact_verifier_when_enabled(monkeypatch):
    llm = FakeLLM({})
    monkeypatch.setattr("medguard.core._build_llm_caller", lambda config: llm)
    config = MedGuardConfig(
        guardrails=GuardrailsConfig(
            phi_detection=PHIConfig(enabled=False),
            drug_safety=DrugSafetyConfig(enabled=False),
            scope_enforcement=ScopeConfig(enabled=False),
            hallucination_detection=HallucinationConfig(enabled=False),
            fact_checking=FactCheckConfig(enabled=True, use_agent=True),
        )
    )

    medguard = MedGuard(config=config)

    assert isinstance(medguard.fact_verifier, AgentFactVerifier)
    assert medguard.fact_verifier._llm_caller is llm
