"""
PubMed fact verification client via NCBI E-utilities.

Used by the FactVerifier guardrail to check medical claims against
peer-reviewed literature. No API key required (3 req/s); set
NCBI_API_KEY for 10 req/s.

Pipeline:
  1. esearch  — find PMIDs matching the claim query
  2. efetch   — retrieve abstracts for top N results
  3. Return structured evidence for scoring
"""
from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger(__name__)

_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


@dataclass
class PubMedArticle:
    pmid: str
    title: str
    abstract: str
    journal: str = ""
    year: str = ""


@dataclass
class FactEvidence:
    claim: str
    supporting: list[PubMedArticle] = field(default_factory=list)
    contradicting: list[PubMedArticle] = field(default_factory=list)
    total_results: int = 0
    verified: bool = False
    confidence: float = 0.0
    summary: str = ""


class PubMedClient:
    """
    Lightweight async PubMed client.

    Rate limits:
      - Without API key: 3 req/s
      - With NCBI_API_KEY env var: 10 req/s
    """

    def __init__(self, http_client, max_results: int = 5) -> None:
        self._http = http_client
        self._max_results = max_results
        self._api_key = os.environ.get("NCBI_API_KEY", "")
        # Rate limiter: conservative 2 req/s without key, 8 with key
        rps = 8.0 if self._api_key else 2.0
        self._semaphore = asyncio.Semaphore(int(rps))

    def _params(self, **kwargs) -> dict:
        p = {"db": "pubmed", **kwargs}
        if self._api_key:
            p["api_key"] = self._api_key
        return p

    async def search(self, query: str) -> list[str]:
        """Search PubMed and return list of PMIDs."""
        async with self._semaphore:
            try:
                r = await self._http.get(
                    _ESEARCH,
                    params=self._params(
                        term=query,
                        retmax=self._max_results,
                        retmode="json",
                        sort="relevance",
                    ),
                    timeout=8.0,
                )
                r.raise_for_status()
                data = r.json()
                return data.get("esearchresult", {}).get("idlist", [])
            except Exception as exc:
                log.debug("pubmed_search_failed", query=query[:60], error=str(exc))
                return []

    async def fetch_summaries(self, pmids: list[str]) -> list[PubMedArticle]:
        """Fetch structured summaries (title, journal, year) for PMIDs."""
        if not pmids:
            return []
        async with self._semaphore:
            try:
                r = await self._http.get(
                    _ESUMMARY,
                    params=self._params(
                        id=",".join(pmids),
                        version="2.0",
                        retmode="json",
                    ),
                    timeout=8.0,
                )
                r.raise_for_status()
                data = r.json()
                articles = []
                result = data.get("result", {})
                for pmid in pmids:
                    doc = result.get(pmid, {})
                    if not doc:
                        continue
                    articles.append(PubMedArticle(
                        pmid=pmid,
                        title=doc.get("title", ""),
                        abstract="",
                        journal=doc.get("source", ""),
                        year=doc.get("pubdate", "")[:4],
                    ))
                return articles
            except Exception as exc:
                log.debug("pubmed_summary_failed", error=str(exc))
                return []

    async def fetch_abstracts(self, pmids: list[str]) -> list[PubMedArticle]:
        """Fetch full abstracts for PMIDs."""
        if not pmids:
            return []
        async with self._semaphore:
            try:
                r = await self._http.get(
                    _EFETCH,
                    params=self._params(
                        id=",".join(pmids),
                        rettype="abstract",
                        retmode="text",
                    ),
                    timeout=10.0,
                )
                r.raise_for_status()
                raw = r.text
                return _parse_text_abstracts(pmids, raw)
            except Exception as exc:
                log.debug("pubmed_fetch_failed", error=str(exc))
                return []

    async def verify_claim(self, claim: str) -> FactEvidence:
        """
        Search PubMed for evidence related to a medical claim.

        Returns FactEvidence with supporting/contradicting articles.
        """
        query = _claim_to_query(claim)
        pmids = await self.search(query)

        if not pmids:
            return FactEvidence(
                claim=claim,
                total_results=0,
                verified=False,
                confidence=0.0,
                summary="No PubMed results found for this claim.",
            )

        # Fetch summaries (light) + abstracts for top 3
        summaries, articles = await asyncio.gather(
            self.fetch_summaries(pmids),
            self.fetch_abstracts(pmids[:3]),
        )

        # Merge abstract text into summaries
        abstract_map = {a.pmid: a.abstract for a in articles}
        for s in summaries:
            s.abstract = abstract_map.get(s.pmid, "")

        evidence = _score_evidence(claim, summaries)
        log.debug(
            "pubmed_claim_verified",
            claim=claim[:60],
            results=len(pmids),
            confidence=evidence.confidence,
        )
        return evidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _claim_to_query(claim: str) -> str:
    """Convert a free-text medical claim into a focused PubMed query."""
    # Strip common filler words, keep medical terms
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "it", "its", "this", "that",
        "for", "with", "and", "or", "but", "in", "on", "at", "to", "of",
        "you", "your", "my", "i", "me", "we", "us", "patient", "patients",
    }
    words = [w for w in re.split(r"\W+", claim.lower()) if w and w not in stopwords]
    # Take top 6 content words
    query = " AND ".join(words[:6])
    return query or claim[:100]


def _parse_text_abstracts(pmids: list[str], raw: str) -> list[PubMedArticle]:
    """Parse NCBI plain-text abstract output into PubMedArticle objects."""
    articles = []
    # Split on PMID blocks — each record starts with \n\n{number}.\n
    blocks = re.split(r"\n\n\d+\.\s", "\n\n" + raw)[1:]

    for i, block in enumerate(blocks):
        pmid = pmids[i] if i < len(pmids) else str(i)
        # Extract title (first non-empty line)
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        title = lines[0] if lines else ""

        # Extract abstract section
        abstract = ""
        in_abstract = False
        for line in lines:
            if line.startswith("Abstract"):
                in_abstract = True
                abstract = line[len("Abstract"):].strip()
            elif in_abstract:
                if re.match(r"^(Author|PMID|DOI|Copyright|©)", line):
                    break
                abstract += " " + line

        articles.append(PubMedArticle(
            pmid=pmid,
            title=title,
            abstract=abstract.strip(),
        ))

    return articles


def _score_evidence(claim: str, articles: list[PubMedArticle]) -> FactEvidence:
    """
    Score articles as supporting or contradicting the claim.

    Simple keyword-based scoring:
    - Presence of negation words near claim terms → contradicting
    - Otherwise → supporting (literature exists on the topic)
    """
    claim_lower = claim.lower()
    claim_words = set(re.split(r"\W+", claim_lower)) - {"the", "a", "is", "and"}

    negation_re = re.compile(
        r"\b(not|no|never|incorrect|false|myth|disproven|contradict|refute|"
        r"does not|do not|cannot|wrong|inaccurate|unsupported)\b",
        re.IGNORECASE,
    )

    supporting = []
    contradicting = []

    for article in articles:
        text = (article.title + " " + article.abstract).lower()
        overlap = sum(1 for w in claim_words if len(w) > 4 and w in text)
        if overlap == 0:
            continue
        if negation_re.search(article.abstract):
            contradicting.append(article)
        else:
            supporting.append(article)

    total = len(articles)
    if total == 0:
        confidence = 0.0
    elif contradicting:
        confidence = max(0.0, 1.0 - (len(contradicting) / total))
    else:
        confidence = min(0.9, 0.5 + (len(supporting) / total) * 0.4)

    verified = confidence >= 0.5 and len(supporting) > 0
    summary_parts = []
    if supporting:
        summary_parts.append(f"{len(supporting)} supporting article(s) found")
    if contradicting:
        summary_parts.append(f"{len(contradicting)} contradicting article(s) found")

    return FactEvidence(
        claim=claim,
        supporting=supporting,
        contradicting=contradicting,
        total_results=total,
        verified=verified,
        confidence=confidence,
        summary="; ".join(summary_parts) or "No relevant abstracts found",
    )
