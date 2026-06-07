"""
PubMed-backed medical fact verification.

Extracts falsifiable medical claims from LLM output and verifies each
against peer-reviewed PubMed literature via NCBI E-utilities.

Claim types detected:
  - Dosage claims    ("metformin max dose is 3000mg")
  - Mechanism claims ("aspirin inhibits COX-2")
  - Safety claims    ("ibuprofen is safe in pregnancy")
  - Drug claims      ("warfarin has a narrow therapeutic index")
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel

if TYPE_CHECKING:
    from medguard.guardrails.protocols import LLMCallerProtocol
    from medguard.knowledge.pubmed import FactEvidence, PubMedClient

log = structlog.get_logger(__name__)


# Patterns that signal a falsifiable medical claim
_CLAIM_PATTERNS = [
    re.compile(
        r"(?:the\s+)?(?:maximum|max|recommended|standard|typical|usual)\s+"
        r"(?:dose|dosage|daily dose)\s+(?:of\s+)?(\w[\w\s]{2,20}?)\s+is\s+"
        r"([\d,\.]+\s*(?:mg|mcg|g|units?))",
        re.IGNORECASE,
    ),
    re.compile(
        r"(\w[\w\s]{2,25}?)\s+(?:is|are|has been|have been)\s+"
        r"(?:shown to|proven to|known to|found to)?\s*"
        r"((?:safe|effective|contraindicated|dangerous|associated with|linked to)"
        r"[\w\s,]{0,50})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(\w[\w\s]{2,20}?)\s+(?:inhibits?|blocks?|activates?|increases?|decreases?|"
        r"reduces?|causes?|prevents?|treats?|cures?)\s+([\w\s]{3,40})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:studies|research|evidence|trials?|data)\s+(?:show|suggest|indicate|"
        r"demonstrate|confirm)\s+(?:that\s+)?([\w\s,]{10,100})",
        re.IGNORECASE,
    ),
]

# Min claim length to avoid trivial matches
_MIN_CLAIM_LEN = 15


class FactCheckResult(BaseModel):
    claims_checked: int
    verified_claims: list[str]
    unverified_claims: list[str]
    low_confidence_claims: list[str]
    overall_confidence: float
    pubmed_evidence: list[dict]  # serialized FactEvidence summaries
    flagged: bool
    annotation: str  # inline [FACT-CHECK: ...] note for the response


class FactVerifier:
    """
    Verifies medical claims in LLM output against PubMed.

    Intended as a post-LLM guardrail stage alongside HallucinationDetector.
    """

    def __init__(self, pubmed: PubMedClient, confidence_threshold: float = 0.4) -> None:
        self._pubmed = pubmed
        self._threshold = confidence_threshold

    async def verify(self, text: str) -> FactCheckResult:
        """Extract and verify all medical claims in text."""
        claims = _extract_claims(text)
        if not claims:
            return FactCheckResult(
                claims_checked=0,
                verified_claims=[],
                unverified_claims=[],
                low_confidence_claims=[],
                overall_confidence=1.0,
                pubmed_evidence=[],
                flagged=False,
                annotation="",
            )

        import asyncio
        evidences: list[FactEvidence] = await asyncio.gather(
            *[self._pubmed.verify_claim(c) for c in claims],
            return_exceptions=True,
        )

        verified = []
        unverified = []
        low_confidence = []
        evidence_summaries = []

        for claim, ev in zip(claims, evidences):
            if isinstance(ev, Exception):
                log.debug("fact_check_error", claim=claim[:50], error=str(ev))
                continue

            evidence_summaries.append({
                "claim": ev.claim,
                "verified": ev.verified,
                "confidence": round(ev.confidence, 2),
                "summary": ev.summary,
                "supporting_pmids": [a.pmid for a in ev.supporting[:3]],
            })

            if ev.total_results == 0:
                unverified.append(claim)
            elif ev.confidence < self._threshold:
                low_confidence.append(claim)
            else:
                verified.append(claim)

        total = len(verified) + len(unverified) + len(low_confidence)
        overall = sum(
            e["confidence"] for e in evidence_summaries
        ) / max(len(evidence_summaries), 1)

        flagged = len(low_confidence) > 0 or len(unverified) > total * 0.5

        annotation = _build_annotation(verified, unverified, low_confidence, evidence_summaries)

        return FactCheckResult(
            claims_checked=len(claims),
            verified_claims=verified,
            unverified_claims=unverified,
            low_confidence_claims=low_confidence,
            overall_confidence=round(overall, 2),
            pubmed_evidence=evidence_summaries,
            flagged=flagged,
            annotation=annotation,
        )


class AgentFactVerifier(FactVerifier):
    """
    Verifies claims by asking an LLM to reason over retrieved PubMed abstracts.

    The agent path is opt-in. If no LLM caller is configured, it falls back to
    the keyword-backed PubMed verifier.
    """

    def __init__(
        self,
        pubmed: PubMedClient,
        llm_caller: LLMCallerProtocol | None,
        confidence_threshold: float = 0.4,
    ) -> None:
        super().__init__(pubmed, confidence_threshold=confidence_threshold)
        self._llm_caller = llm_caller

    async def verify_claim(self, claim: str) -> FactEvidence:
        """Return agent-scored PubMed evidence for one claim."""
        if self._llm_caller is None:
            return await self._pubmed.verify_claim(claim)

        pmids = await self._pubmed.search(claim)
        if not pmids:
            return await self._pubmed.verify_claim(claim)

        import asyncio

        summaries, abstracts = await asyncio.gather(
            self._pubmed.fetch_summaries(pmids),
            self._pubmed.fetch_abstracts(pmids[:5]),
        )
        abstract_map = {article.pmid: article.abstract for article in abstracts}
        articles = []
        for summary in summaries:
            summary.abstract = abstract_map.get(summary.pmid, "")
            articles.append(summary)

        if not articles:
            return await self._pubmed.verify_claim(claim)

        prompt = _build_agent_prompt(claim, articles[:5])
        try:
            raw_response = await self._llm_caller.call(prompt)
            verdict = _parse_agent_verdict(raw_response)
        except Exception as exc:
            log.debug("agent_fact_check_failed", claim=claim[:50], error=str(exc))
            return await self._pubmed.verify_claim(claim)

        cited_pmids = set(verdict.get("citations", []))
        cited_articles = [article for article in articles if article.pmid in cited_pmids]
        evidence_articles = cited_articles or articles[:3]
        verdict_name = str(verdict.get("verdict", "inconclusive")).lower()
        confidence = _normalize_confidence(verdict.get("confidence", 0.0))
        reasoning = str(verdict.get("reasoning", "")).strip()

        from medguard.knowledge.pubmed import FactEvidence

        return FactEvidence(
            claim=claim,
            supporting=evidence_articles if verdict_name == "supported" else [],
            contradicting=evidence_articles if verdict_name == "contradicted" else [],
            total_results=len(articles),
            verified=verdict_name == "supported" and confidence >= self._threshold,
            confidence=confidence,
            summary=f"Agent verdict: {verdict_name}",
            reasoning=reasoning,
        )

    async def verify(self, text: str) -> FactCheckResult:
        """Extract claims and verify each with agent reasoning over PubMed."""
        claims = _extract_claims(text)
        if not claims:
            return FactCheckResult(
                claims_checked=0,
                verified_claims=[],
                unverified_claims=[],
                low_confidence_claims=[],
                overall_confidence=1.0,
                pubmed_evidence=[],
                flagged=False,
                annotation="",
            )

        import asyncio

        evidences: list[FactEvidence] = await asyncio.gather(
            *[self.verify_claim(c) for c in claims],
            return_exceptions=True,
        )

        verified = []
        unverified = []
        low_confidence = []
        evidence_summaries = []

        for claim, ev in zip(claims, evidences):
            if isinstance(ev, Exception):
                log.debug("agent_fact_check_error", claim=claim[:50], error=str(ev))
                continue

            evidence_summaries.append({
                "claim": ev.claim,
                "verified": ev.verified,
                "confidence": round(ev.confidence, 2),
                "summary": ev.summary,
                "reasoning": ev.reasoning,
                "supporting_pmids": [a.pmid for a in ev.supporting[:3]],
                "contradicting_pmids": [a.pmid for a in ev.contradicting[:3]],
            })

            if ev.total_results == 0:
                unverified.append(claim)
            elif ev.confidence < self._threshold:
                low_confidence.append(claim)
            else:
                verified.append(claim)

        total = len(verified) + len(unverified) + len(low_confidence)
        overall = sum(
            e["confidence"] for e in evidence_summaries
        ) / max(len(evidence_summaries), 1)
        flagged = len(low_confidence) > 0 or len(unverified) > total * 0.5
        annotation = _build_annotation(verified, unverified, low_confidence, evidence_summaries)

        return FactCheckResult(
            claims_checked=len(claims),
            verified_claims=verified,
            unverified_claims=unverified,
            low_confidence_claims=low_confidence,
            overall_confidence=round(overall, 2),
            pubmed_evidence=evidence_summaries,
            flagged=flagged,
            annotation=annotation,
        )


def _extract_claims(text: str) -> list[str]:
    """Extract falsifiable medical claims from text using regex patterns."""
    seen: set[str] = set()
    claims = []

    for pattern in _CLAIM_PATTERNS:
        for match in pattern.finditer(text):
            claim = match.group(0).strip()
            # Deduplicate and filter short/trivial claims
            normalized = re.sub(r"\s+", " ", claim.lower())
            if len(claim) >= _MIN_CLAIM_LEN and normalized not in seen:
                seen.add(normalized)
                claims.append(claim)

    return claims[:8]  # cap at 8 to avoid excessive API calls


def _build_agent_prompt(claim: str, articles: list) -> str:
    evidence = []
    for article in articles:
        abstract = article.abstract or article.title
        evidence.append(
            f"PMID: {article.pmid}\nTitle: {article.title}\nAbstract: {abstract[:1200]}"
        )
    return (
        "You are verifying a medical claim against retrieved PubMed abstracts.\n"
        "Return only JSON with keys: verdict, confidence, citations, reasoning.\n"
        "verdict must be one of: supported, contradicted, inconclusive.\n"
        "confidence must be a number from 0 to 1.\n"
        "citations must contain only PMIDs from the provided evidence.\n\n"
        f"Claim: {claim}\n\n"
        "Evidence:\n" + "\n\n".join(evidence)
    )


def _parse_agent_verdict(raw_response: str) -> dict:
    raw_response = raw_response.strip()
    if raw_response.startswith("```"):
        raw_response = re.sub(r"^```(?:json)?\s*", "", raw_response)
        raw_response = re.sub(r"\s*```$", "", raw_response)
    parsed = json.loads(raw_response)
    verdict = str(parsed.get("verdict", "inconclusive")).lower()
    if verdict not in {"supported", "contradicted", "inconclusive"}:
        parsed["verdict"] = "inconclusive"
    citations = parsed.get("citations", [])
    parsed["citations"] = [str(citation) for citation in citations if citation]
    return parsed


def _normalize_confidence(value) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _build_annotation(
    verified: list[str],
    unverified: list[str],
    low_confidence: list[str],
    evidence: list[dict],
) -> str:
    if not (unverified or low_confidence):
        return ""

    parts = []
    if low_confidence:
        parts.append(
            f"Low PubMed evidence for: {'; '.join(c[:60] for c in low_confidence[:2])}"
        )
    if unverified:
        parts.append(
            f"No PubMed results for: {'; '.join(c[:60] for c in unverified[:2])}"
        )

    pmids = [p for e in evidence for p in e.get("supporting_pmids", [])[:1]]
    if pmids:
        parts.append(f"See PMIDs: {', '.join(pmids[:3])}")

    return f"[FACT-CHECK: {' | '.join(parts)}]" if parts else ""
