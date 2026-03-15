"""
SNOMED-CT concept validation.

Primary lookup: bundled JSON file (~15K most common clinical concepts).
Fallback: IHTSDO Snowstorm browser API (rate-limited, used sparingly).

The bundled concept file covers all ICD-10-mapped conditions, common drugs,
anatomy terms, and procedures. It ships inside the wheel at:
    medguard/knowledge/data/snomed_concepts_subset.json

Format: {"myocardial infarction": "22298006", "type 2 diabetes mellitus": "44054006", ...}
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import structlog

log = structlog.get_logger(__name__)

_SNOWSTORM_URL = (
    "https://browser.ihtsdotools.org/snowstorm/snomed-ct/browser/MAIN/concepts"
)
_DATA_FILE = Path(__file__).parent / "data" / "snomed_concepts_subset.json"


class SNOMEDRateLimitError(Exception):
    pass


class SNOMEDClient:
    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._http = http_client
        self._semaphore = asyncio.Semaphore(2)
        self._concepts: dict[str, str] = {}
        self._loaded = False

    def _load_bundled_concepts(self) -> dict[str, str]:
        if not _DATA_FILE.exists():
            log.warning("snomed_bundle_not_found", path=str(_DATA_FILE))
            return {}
        with _DATA_FILE.open() as f:
            return json.load(f)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._concepts = self._load_bundled_concepts()
            self._loaded = True

    def is_valid_concept(self, term: str) -> bool:
        """
        Check if a term is a recognized SNOMED-CT concept.
        Uses bundled dict first; Snowstorm is not called from sync context.
        """
        self._ensure_loaded()
        term_lower = term.lower().strip()
        # Exact match
        if term_lower in self._concepts:
            return True
        # Substring match (e.g., "chest pain" found in "chest pain on exertion")
        return any(term_lower in concept for concept in self._concepts)

    async def find_concepts(self, term: str, limit: int = 5) -> list[dict]:
        """
        Find SNOMED concepts matching a term.
        Checks bundled data first; falls back to Snowstorm API.
        """
        self._ensure_loaded()
        term_lower = term.lower().strip()

        # Check bundled data
        bundled_matches = [
            {"term": k, "conceptId": v, "source": "bundled"}
            for k, v in self._concepts.items()
            if term_lower in k.lower()
        ][:limit]

        if bundled_matches:
            return bundled_matches

        # Fallback to Snowstorm
        return await self._query_snowstorm(term, limit)

    async def _query_snowstorm(self, term: str, limit: int) -> list[dict]:
        async with self._semaphore:
            try:
                r = await self._http.get(
                    _SNOWSTORM_URL,
                    params={
                        "term": term,
                        "limit": limit,
                        "activeFilter": "true",
                    },
                    timeout=5.0,
                )
                if r.status_code == 429:
                    raise SNOMEDRateLimitError("SNOMED Snowstorm API rate limit exceeded")
                r.raise_for_status()
                data = r.json()
                items = data.get("items", [])
                return [
                    {
                        "term": item.get("pt", {}).get("term", ""),
                        "conceptId": item.get("conceptId", ""),
                        "source": "snowstorm",
                    }
                    for item in items
                ]
            except SNOMEDRateLimitError:
                raise
            except Exception as exc:
                log.debug("snowstorm_error", error=str(exc), term=term)
                return []
