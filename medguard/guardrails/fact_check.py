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

import re
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel

if TYPE_CHECKING:
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
