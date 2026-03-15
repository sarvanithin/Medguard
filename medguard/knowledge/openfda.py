"""
OpenFDA API client for drug interaction and adverse event data.

Primary data source for drug-drug interactions and contraindications.
The NLM Drug Interaction API was discontinued Jan 2024; OpenFDA Drug Label
API provides the same data via the FDA-approved label text.

Endpoints used:
  - Drug label: https://api.fda.gov/drug/label.json
  - Adverse events: https://api.fda.gov/drug/event.json

Rate limits: 240 requests/minute (no key), 1000/minute (with key).
We use asyncio.Semaphore(4) to stay well within free tier limits.
"""
from __future__ import annotations

import asyncio
import re
from enum import Enum
from pathlib import Path
from typing import Literal

import httpx
import structlog
from pydantic import BaseModel

from medguard.config import DrugSafetyConfig

log = structlog.get_logger(__name__)

_LABEL_URL = "https://api.fda.gov/drug/label.json"
_EVENT_URL = "https://api.fda.gov/drug/event.json"


class InteractionSeverity(str, Enum):
    CONTRAINDICATED = "contraindicated"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNKNOWN = "unknown"


class DrugInteraction(BaseModel):
    drug_a: str
    drug_b: str
    severity: InteractionSeverity
    description: str
    source: Literal["openfda_label", "static_table", "rxnorm_class"]
    confidence: float = 0.8


class ContraindicationResult(BaseModel):
    drug: str
    condition: str
    description: str


# Severity signal patterns (applied to FDA label text)
_CONTRAINDICATED_RE = re.compile(
    r"\b(contraindicated|do not (use|administer|take)|must not (use|be used))\b",
    re.IGNORECASE,
)
_HIGH_RE = re.compile(
    r"\b(avoid|serious|life[- ]threatening|fatal|death|severe|"
    r"do not co-?administer|should not be used (together|concomitantly)|"
    r"significant(ly)? (increase|elevated|potentiate)|marked(ly)?|"
    r"substantially|greatly (increase|elevate)|risk of (bleeding|hemorrhage|death|toxicity))\b",
    re.IGNORECASE,
)
_MODERATE_RE = re.compile(
    r"\b(monitor|caution|adjust (dose|dosage)|use with caution|"
    r"close monitoring|clinical monitoring|may (increase|decrease|affect|alter)|"
    r"prolongation|bleeding time|prothrombin time|INR|can (increase|decrease|displace|inhibit)|"
    r"displac|protein binding|clearance|interact)\b",
    re.IGNORECASE,
)
_LOW_RE = re.compile(
    r"\b(minor|small (increase|decrease)|may slightly|"
    r"generally (not|well) tolerated)\b",
    re.IGNORECASE,
)

# Contraindication condition patterns
_CONDITION_RE = re.compile(
    r"(?:contraindicated in|should not be used in|not recommended in)\s+"
    r"([^.;]{5,80})",
    re.IGNORECASE,
)


class OpenFDAClient:
    def __init__(self, config: DrugSafetyConfig, http_client: httpx.AsyncClient) -> None:
        self.config = config
        self._http = http_client
        self._semaphore = asyncio.Semaphore(4)
        self._cache = _build_cache(config.cache_ttl_seconds)

    async def get_drug_interactions(
        self, drug_a: str, drug_b: str
    ) -> DrugInteraction | None:
        """
        Check for interaction between two drugs using FDA label data.

        Queries the drug_interactions field of drug_a's label for mentions
        of drug_b, then parses severity from the text.
        """
        cache_key = f"interaction:{drug_a.lower()}:{drug_b.lower()}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return DrugInteraction(**cached) if cached else None

        async with self._semaphore:
            try:
                r = await self._http.get(
                    _LABEL_URL,
                    params={
                        "search": (
                            f'openfda.generic_name:"{drug_a}"'
                            f' AND drug_interactions:"{drug_b}"'
                        ),
                        "limit": "3",
                    },
                    timeout=self.config.api_timeout_seconds,
                )

                if r.status_code == 404:
                    self._cache.set(cache_key, None, expire=self.config.cache_ttl_seconds)
                    return None

                r.raise_for_status()
                data = r.json()
                results = data.get("results", [])

                if not results:
                    self._cache.set(cache_key, None, expire=self.config.cache_ttl_seconds)
                    return None

                # Extract the most relevant interaction text
                interaction_text = ""
                for result in results:
                    texts = result.get("drug_interactions", [])
                    if texts:
                        # Find the paragraph mentioning drug_b
                        combined = " ".join(texts)
                        drug_b_lower = drug_b.lower()
                        for sentence in combined.split("."):
                            if drug_b_lower in sentence.lower():
                                interaction_text += sentence.strip() + ". "
                        if not interaction_text:
                            interaction_text = combined[:500]
                        break

                if not interaction_text:
                    self._cache.set(cache_key, None, expire=self.config.cache_ttl_seconds)
                    return None

                severity = self._parse_severity(interaction_text)
                interaction = DrugInteraction(
                    drug_a=drug_a,
                    drug_b=drug_b,
                    severity=severity,
                    description=interaction_text.strip()[:400],
                    source="openfda_label",
                )
                self._cache.set(
                    cache_key,
                    interaction.model_dump(),
                    expire=self.config.cache_ttl_seconds,
                )
                return interaction

            except httpx.HTTPStatusError as exc:
                log.warning(
                    "openfda_http_error",
                    status=exc.response.status_code,
                    drug_a=drug_a,
                    drug_b=drug_b,
                )
                return None
            except Exception as exc:
                log.warning("openfda_interaction_error", error=str(exc))
                return None

    async def get_contraindications(self, drug: str) -> list[ContraindicationResult]:
        """
        Return contraindicated conditions for a drug from FDA label text.
        """
        cache_key = f"contra:{drug.lower()}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return [ContraindicationResult(**c) for c in cached] if cached else []

        async with self._semaphore:
            try:
                r = await self._http.get(
                    _LABEL_URL,
                    params={
                        "search": f'openfda.generic_name:"{drug}"',
                        "limit": "1",
                    },
                    timeout=self.config.api_timeout_seconds,
                )
                if r.status_code == 404:
                    return []
                r.raise_for_status()
                data = r.json()
                results = data.get("results", [])
                if not results:
                    return []

                contra_texts = results[0].get("contraindications", [])
                boxed = results[0].get("boxed_warning", [])
                all_text = " ".join(contra_texts + boxed)

                conditions = []
                for match in _CONDITION_RE.finditer(all_text):
                    condition_text = match.group(1).strip().rstrip(",")
                    conditions.append(
                        ContraindicationResult(
                            drug=drug,
                            condition=condition_text[:100],
                            description=match.group(0)[:200],
                        )
                    )

                self._cache.set(
                    cache_key,
                    [c.model_dump() for c in conditions],
                    expire=self.config.cache_ttl_seconds,
                )
                return conditions

            except Exception as exc:
                log.debug("openfda_contraindication_error", error=str(exc), drug=drug)
                return []

    async def get_adverse_event_count(
        self, drug: str, serious_only: bool = True
    ) -> int:
        """
        Return the total count of adverse event reports for a drug.
        Used as a severity proxy signal.
        """
        async with self._semaphore:
            try:
                search = f'patient.drug.openfda.generic_name:"{drug}"'
                if serious_only:
                    search += " AND serious:1"
                r = await self._http.get(
                    _EVENT_URL,
                    params={"search": search, "limit": "1"},
                    timeout=self.config.api_timeout_seconds,
                )
                if r.status_code == 404:
                    return 0
                r.raise_for_status()
                data = r.json()
                return data.get("meta", {}).get("results", {}).get("total", 0)
            except Exception:
                return 0

    def _parse_severity(self, text: str) -> InteractionSeverity:
        """
        Extract severity signal from FDA label interaction text.
        Rules applied in priority order (highest severity first).
        """
        if _CONTRAINDICATED_RE.search(text):
            return InteractionSeverity.CONTRAINDICATED
        if _HIGH_RE.search(text):
            return InteractionSeverity.HIGH
        if _MODERATE_RE.search(text):
            return InteractionSeverity.MODERATE
        if _LOW_RE.search(text):
            return InteractionSeverity.LOW
        return InteractionSeverity.UNKNOWN


def _build_cache(ttl_seconds: int):
    try:
        import diskcache
        cache_dir = Path.home() / ".medguard" / "cache" / "openfda"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return diskcache.Cache(str(cache_dir))
    except ImportError:
        from medguard.knowledge.rxnorm import _MemoryCache
        return _MemoryCache(ttl_seconds)
