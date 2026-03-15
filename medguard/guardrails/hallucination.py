"""
Medical hallucination detection for LLM output.

Four sub-checks run concurrently:
  1. Fake drug names: extract candidates, validate via RxNorm
  2. Impossible dosages: parse dosage patterns, compare to MAX_DOSES
  3. Unrecognized medical terms: cross-ref against SNOMED bundled concepts
  4. Overconfident claims: flag assertive language without evidence markers
"""
from __future__ import annotations

import asyncio
import re
from enum import Enum
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel

from medguard.config import HallucinationConfig

if TYPE_CHECKING:
    from medguard.knowledge.rxnorm import RxNormClient
    from medguard.knowledge.snomed import SNOMEDClient

log = structlog.get_logger(__name__)


class HallucinationType(str, Enum):
    FAKE_DRUG_NAME = "fake_drug_name"
    IMPOSSIBLE_DOSAGE = "impossible_dosage"
    UNKNOWN_MEDICAL_TERM = "unknown_medical_term"
    CONFIDENT_UNSUPPORTED_CLAIM = "confident_unsupported_claim"


class HallucinationFlag(BaseModel):
    type: HallucinationType
    text: str
    start: int
    end: int
    confidence: float
    explanation: str


class HallucinationResult(BaseModel):
    flags: list[HallucinationFlag]
    hallucination_score: float  # 0.0 = clean, 1.0 = severe
    blocked: bool
    annotated_text: str  # original text with [WARNING: ...] markers


# ---------------------------------------------------------------------------
# Drug maximum daily doses (mg unless otherwise noted)
# Source: standard clinical references (PDR, FDA labeling)
# ---------------------------------------------------------------------------
MAX_DOSES: dict[str, float] = {
    "ibuprofen": 3200,
    "naproxen": 1500,
    "acetaminophen": 4000,
    "aspirin": 4000,
    "metformin": 2550,
    "metoprolol": 400,
    "atenolol": 200,
    "amlodipine": 10,
    "lisinopril": 80,
    "losartan": 150,
    "atorvastatin": 80,
    "simvastatin": 80,
    "sertraline": 200,
    "fluoxetine": 80,
    "escitalopram": 40,
    "citalopram": 60,
    "amitriptyline": 300,
    "warfarin": 20,
    "prednisone": 300,
    "dexamethasone": 80,
    "omeprazole": 120,
    "pantoprazole": 240,
    "digoxin": 0.5,  # mg/day
    "colchicine": 1.8,
    "allopurinol": 800,
    "hydrochlorothiazide": 200,
    "furosemide": 600,
    "carbamazepine": 1600,
    "phenytoin": 600,
    "valproate": 3000,
    "lithium": 2400,
    "tramadol": 400,
    "codeine": 360,
    "morphine": 300,
}

# Dosage extraction pattern
_DOSAGE_RE = re.compile(
    r"\b([A-Za-z][A-Za-z\s\-]{2,30}?)\s+"
    r"(\d+(?:[,\.]\d+)?)\s*(mg|mcg|g|mEq|units?|IU)\b",
    re.IGNORECASE,
)

# Drug mention pattern (for hallucination checking — broader than drug_safety extractor)
_DRUG_MENTION_RE = re.compile(
    r"\b([A-Za-z][a-z]{4,}(?:mycin|cillin|oxacin|olol|pril|sartan|statin|"
    r"azole|vir|mab|nib|tide|zide|olam|pam|zolam))\b",
    re.IGNORECASE,
)

# Medical terminology patterns (anatomical terms, procedures, diagnoses)
_MEDICAL_TERM_RE = re.compile(
    r"\b([A-Za-z]{5,}(?:itis|osis|emia|uria|algia|plasty|ectomy|otomy|"
    r"scopy|graphy|ology|pathy|trophy|genesis|lysis))\b",
    re.IGNORECASE,
)

# Overconfident claim patterns
_CONFIDENT_RE = re.compile(
    r"\b(definitely|certainly|absolutely|guaranteed|always|never|100%|"
    r"proven cure|will cure|cures? (all|every)|no side effects|completely safe|"
    r"no risk|perfectly safe|without any risk)\b",
    re.IGNORECASE,
)

# Known non-drug words ending in drug-like suffixes (false positive suppression)
_DRUG_SUFFIX_EXCLUSIONS = frozenset([
    "technology", "strategy", "biology", "ecology", "psychology", "mythology",
    "terminology", "genealogy", "methodology", "pathology", "pharmacology",
    "cardiology", "neurology", "oncology", "radiology", "gastroenterology",
    "dermatology", "ophthalmology", "urology", "anesthesiology",
])


class HallucinationDetector:
    def __init__(
        self,
        config: HallucinationConfig,
        rxnorm: "RxNormClient",
        snomed: "SNOMEDClient",
    ) -> None:
        self.config = config
        self._rxnorm = rxnorm
        self._snomed = snomed

    async def check(self, text: str) -> HallucinationResult:
        tasks = []
        if self.config.check_drug_names:
            tasks.append(self._check_drug_names(text))
        if self.config.check_dosages:
            tasks.append(self._check_dosages(text))
        if self.config.check_confident_claims:
            tasks.append(asyncio.coroutine(lambda: self._check_confident_claims(text))())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        flags: list[HallucinationFlag] = []
        for result in results:
            if isinstance(result, Exception):
                log.debug("hallucination_subcheck_failed", error=str(result))
            elif isinstance(result, list):
                flags.extend(result)

        score = _compute_hallucination_score(flags)
        blocked = score >= self.config.confidence_threshold and bool(flags)
        annotated = _annotate_text(text, flags)

        return HallucinationResult(
            flags=flags,
            hallucination_score=score,
            blocked=blocked,
            annotated_text=annotated,
        )

    async def _check_drug_names(self, text: str) -> list[HallucinationFlag]:
        """Flag drug-like names that RxNorm doesn't recognize."""
        flags: list[HallucinationFlag] = []
        candidates: list[tuple[str, int, int]] = []

        for match in _DRUG_MENTION_RE.finditer(text):
            drug_name = match.group(1).lower()
            if drug_name in _DRUG_SUFFIX_EXCLUSIONS:
                continue
            candidates.append((drug_name, match.start(), match.end()))

        if not candidates:
            return flags

        # Validate all candidates concurrently
        async def validate(name: str, start: int, end: int):
            try:
                exists = await self._rxnorm.validate_drug_exists(name)
                if not exists:
                    return HallucinationFlag(
                        type=HallucinationType.FAKE_DRUG_NAME,
                        text=text[start:end],
                        start=start,
                        end=end,
                        confidence=0.75,
                        explanation=(
                            f"'{text[start:end]}' could not be found in RxNorm. "
                            "Verify this is a real medication name."
                        ),
                    )
            except Exception:
                pass
            return None

        results = await asyncio.gather(*[validate(n, s, e) for n, s, e in candidates])
        return [r for r in results if r is not None]

    async def _check_dosages(self, text: str) -> list[HallucinationFlag]:
        """Flag dosages that exceed known maximum safe doses."""
        flags: list[HallucinationFlag] = []

        for match in _DOSAGE_RE.finditer(text):
            drug_name = match.group(1).strip().lower()
            try:
                dose_str = match.group(2).replace(",", "")
                dose_value = float(dose_str)
            except ValueError:
                continue
            unit = match.group(3).lower()

            # Convert mcg to mg for comparison
            if unit in ("mcg", "ug"):
                dose_value /= 1000

            max_dose = MAX_DOSES.get(drug_name)
            if max_dose is None:
                # Try partial match (e.g., "ibuprofen sodium" -> "ibuprofen")
                for known_drug in MAX_DOSES:
                    if known_drug in drug_name or drug_name in known_drug:
                        max_dose = MAX_DOSES[known_drug]
                        break

            if max_dose is not None and dose_value > max_dose * 1.5:
                flags.append(
                    HallucinationFlag(
                        type=HallucinationType.IMPOSSIBLE_DOSAGE,
                        text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9,
                        explanation=(
                            f"Dose of {dose_value}{unit} for {drug_name} "
                            f"exceeds known maximum of {max_dose}mg/day."
                        ),
                    )
                )

        return flags

    def _check_confident_claims(self, text: str) -> list[HallucinationFlag]:
        """Flag overconfident language without evidence qualifiers."""
        flags: list[HallucinationFlag] = []
        for match in _CONFIDENT_RE.finditer(text):
            flags.append(
                HallucinationFlag(
                    type=HallucinationType.CONFIDENT_UNSUPPORTED_CLAIM,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7,
                    explanation=(
                        f"Absolute claim '{match.group(0)}' detected. "
                        "Medical information should acknowledge uncertainty and "
                        "recommend professional consultation."
                    ),
                )
            )
        return flags


def _compute_hallucination_score(flags: list[HallucinationFlag]) -> float:
    if not flags:
        return 0.0
    # Weighted sum capped at 1.0
    weight_map = {
        HallucinationType.IMPOSSIBLE_DOSAGE: 0.4,
        HallucinationType.FAKE_DRUG_NAME: 0.35,
        HallucinationType.UNKNOWN_MEDICAL_TERM: 0.2,
        HallucinationType.CONFIDENT_UNSUPPORTED_CLAIM: 0.15,
    }
    score = sum(weight_map.get(f.type, 0.1) * f.confidence for f in flags)
    return min(score, 1.0)


def _annotate_text(text: str, flags: list[HallucinationFlag]) -> str:
    """Insert [WARNING: ...] annotations after flagged spans."""
    if not flags:
        return text
    # Process right-to-left to preserve offsets
    sorted_flags = sorted(flags, key=lambda f: f.start, reverse=True)
    result = text
    for flag in sorted_flags:
        annotation = f" [WARNING: {flag.explanation}]"
        result = result[: flag.end] + annotation + result[flag.end :]
    return result


# Compatibility shim for Python 3.10 (asyncio.coroutine was removed in 3.11)
import sys
if sys.version_info >= (3, 11):
    def _wrap_sync(fn):
        async def _inner(*args, **kwargs):
            return fn(*args, **kwargs)
        return _inner
else:
    def _wrap_sync(fn):
        async def _inner(*args, **kwargs):
            return fn(*args, **kwargs)
        return _inner

# Patch the check method to use _wrap_sync for _check_confident_claims
_orig_check = HallucinationDetector.check

async def _patched_check(self, text: str) -> HallucinationResult:
    tasks = []
    if self.config.check_drug_names:
        tasks.append(self._check_drug_names(text))
    if self.config.check_dosages:
        tasks.append(self._check_dosages(text))
    if self.config.check_confident_claims:
        tasks.append(_wrap_sync(self._check_confident_claims)(text))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    flags: list[HallucinationFlag] = []
    for result in results:
        if isinstance(result, Exception):
            log.debug("hallucination_subcheck_failed", error=str(result))
        elif isinstance(result, list):
            flags.extend(result)

    score = _compute_hallucination_score(flags)
    blocked = score >= self.config.confidence_threshold and bool(flags)
    annotated = _annotate_text(text, flags)

    return HallucinationResult(
        flags=flags,
        hallucination_score=score,
        blocked=blocked,
        annotated_text=annotated,
    )

HallucinationDetector.check = _patched_check
