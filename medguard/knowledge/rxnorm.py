"""
RxNorm API client — drug name normalization only.

The NLM Drug Interaction API was discontinued in Jan 2024.
This client uses only the live endpoints for:
  - Drug name → RxCUI lookup
  - RxCUI → canonical drug name
  - Drug class lookup (for class-level interaction detection)
  - Drug existence validation (for hallucination detection)

All responses are cached with diskcache to minimize API calls.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from medguard.config import DrugSafetyConfig

log = structlog.get_logger(__name__)

_BASE_URL = "https://rxnav.nlm.nih.gov/REST"


class RxNormClient:
    def __init__(self, config: DrugSafetyConfig, http_client: httpx.AsyncClient) -> None:
        self.config = config
        self._http = http_client
        self._cache = _build_cache(config.cache_ttl_seconds)
        self._semaphore = asyncio.Semaphore(5)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=5))
    async def get_rxcui(self, drug_name: str) -> str | None:
        """
        Normalize a drug name to its canonical RxCUI.

        Returns None if the drug is not found in RxNorm (likely a hallucination
        or misspelling).
        """
        key = f"rxcui:{drug_name.lower().strip()}"
        if key in self._cache:
            return self._cache[key]

        async with self._semaphore:
            try:
                r = await self._http.get(
                    f"{_BASE_URL}/rxcui.json",
                    params={"name": drug_name, "search": "1"},
                    timeout=self.config.api_timeout_seconds,
                )
                r.raise_for_status()
                data = r.json()
                rxcui = data.get("idGroup", {}).get("rxnormId", [None])[0]
                self._cache.set(key, rxcui, expire=self.config.cache_ttl_seconds)
                return rxcui
            except httpx.HTTPStatusError as exc:
                log.warning("rxnorm_http_error", status=exc.response.status_code, drug=drug_name)
                return None
            except Exception as exc:
                log.warning("rxnorm_error", error=str(exc), drug=drug_name)
                raise

    async def normalize_drug_name(self, rxcui: str) -> str | None:
        """Return the canonical generic name for a given RxCUI."""
        key = f"name:{rxcui}"
        if key in self._cache:
            return self._cache[key]

        async with self._semaphore:
            try:
                r = await self._http.get(
                    f"{_BASE_URL}/rxcui/{rxcui}/properties.json",
                    timeout=self.config.api_timeout_seconds,
                )
                r.raise_for_status()
                data = r.json()
                name = data.get("properties", {}).get("name")
                self._cache.set(key, name, expire=self.config.cache_ttl_seconds)
                return name
            except Exception as exc:
                log.warning("rxnorm_properties_error", error=str(exc), rxcui=rxcui)
                return None

    async def get_drug_classes(self, rxcui: str) -> list[str]:
        """
        Return drug class names for a given RxCUI.
        Used for class-level interaction detection (e.g., NSAID + anticoagulant).
        """
        key = f"classes:{rxcui}"
        if key in self._cache:
            return self._cache[key]

        async with self._semaphore:
            try:
                r = await self._http.get(
                    f"{_BASE_URL}/rxcui/{rxcui}/related.json",
                    params={"rela": "isa"},
                    timeout=self.config.api_timeout_seconds,
                )
                r.raise_for_status()
                data = r.json()
                concepts = (
                    data.get("relatedGroup", {})
                    .get("conceptGroup", [{}])[0]
                    .get("conceptProperties", [])
                )
                classes = [c.get("name", "") for c in concepts if c.get("name")]
                self._cache.set(key, classes, expire=self.config.cache_ttl_seconds)
                return classes
            except Exception as exc:
                log.debug("rxnorm_classes_error", error=str(exc), rxcui=rxcui)
                return []

    async def validate_drug_exists(self, drug_name: str) -> bool:
        """
        Returns True if RxNorm recognizes the drug name.
        Used by the hallucination detector to flag fabricated drug names.
        """
        rxcui = await self.get_rxcui(drug_name)
        return rxcui is not None


def _build_cache(ttl_seconds: int):
    """Build a persistent disk cache at ~/.medguard/cache/rxnorm/."""
    try:
        import diskcache

        cache_dir = Path.home() / ".medguard" / "cache" / "rxnorm"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return diskcache.Cache(str(cache_dir))
    except ImportError:
        # Fallback to in-memory dict if diskcache not available
        log.warning("diskcache_not_available_using_memory_cache")
        return _MemoryCache(ttl_seconds)


class _MemoryCache:
    """Minimal in-memory cache fallback."""

    def __init__(self, ttl: int) -> None:
        self._store: dict = {}
        self._ttl = ttl

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __getitem__(self, key: str):
        return self._store[key]

    def set(self, key: str, value, expire: int = 0) -> None:
        self._store[key] = value
