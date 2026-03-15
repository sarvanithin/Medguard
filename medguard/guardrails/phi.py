"""
PHI (Protected Health Information) detection and redaction.

Tier 1 (default, zero-dependency): RegexPHIEngine
    Detects SSN, DOB, phone, email, MRN, ZIP, and labeled names.
    NAME detection requires labeled context ("Patient: John Smith") to
    avoid high false-positive rates from free-form NER.

Tier 2 (optional, pip install medguard[nlp]): PresidioPHIEngine
    Uses Microsoft Presidio + spaCy for higher recall including free-form names.

Tier 3 (optional, pip install medguard[aws]): AWSComprehendPHIEngine
    Uses AWS Comprehend Medical for maximum recall in production.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from medguard.config import PHIConfig


@dataclass
class PHIMatch:
    entity_type: str
    start: int
    end: int
    text: str
    confidence: float
    redacted_text: str


@dataclass
class PHIResult:
    original: str
    processed: str
    matches: list[PHIMatch]
    phi_detected: bool
    engine_used: str


# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_SSN_UNLABELED = re.compile(r"\b\d{3}\s\d{2}\s\d{4}\b")

_PHONE = re.compile(
    r"\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

_DOB = re.compile(
    r"\b(0?[1-9]|1[0-2])[/\-](0?[1-9]|[12]\d|3[01])[/\-](19|20)\d{2}\b"
)
_DOB_LONG = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s+(19|20)\d{2}\b",
    re.IGNORECASE,
)

# Medical Record Number — explicitly labeled
_MRN = re.compile(r"\bMRN[:\s#]*\d{5,10}\b", re.IGNORECASE)
_NPI = re.compile(r"\bNPI[:\s#]*\d{10}\b", re.IGNORECASE)
_DEA = re.compile(r"\b[A-Z]{2}\d{7}\b")  # DEA number format

_ZIP = re.compile(r"\b\d{5}(?:-\d{4})?\b")

_ADDRESS = re.compile(
    r"\b\d{1,5}\s+[A-Za-z0-9\s,\.]+(?:Street|St|Avenue|Ave|Boulevard|Blvd|"
    r"Road|Rd|Drive|Dr|Lane|Ln|Court|Ct|Way|Place|Pl)\b",
    re.IGNORECASE,
)

# Names only when preceded by clear label tokens AND the name is Title Case.
# Using inline (?i:...) for the label part only — the name itself must be Title Case
# to avoid false positives on "Patient takes aspirin" or "patient at john.doe@..."
_NAME_LABELED = re.compile(
    r"(?i:Patient|Name|Client|Pt|DOB of|Name of)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
)


_PATTERN_MAP: dict[str, list[re.Pattern]] = {
    "SSN": [_SSN, _SSN_UNLABELED],
    "PHONE": [_PHONE],
    "EMAIL": [_EMAIL],
    "DOB": [_DOB, _DOB_LONG],
    "MRN": [_MRN],
    "NPI": [_NPI],
    "DEA": [_DEA],
    "ZIP": [_ZIP],
    "ADDRESS": [_ADDRESS],
    "NAME_LABELED": [_NAME_LABELED],
}


class RegexPHIEngine:
    """
    Zero-dependency regex-based PHI detection engine.

    Included in the default install. Covers all HIPAA Safe Harbor
    identifiers detectable via pattern matching. Free-form name detection
    requires labeled context tokens to maintain acceptable false-positive rates.
    """

    def analyze(self, text: str, entities: list[str]) -> list[PHIMatch]:
        matches: list[PHIMatch] = []
        for entity in entities:
            patterns = _PATTERN_MAP.get(entity, [])
            for pattern in patterns:
                for m in pattern.finditer(text):
                    # For NAME_LABELED, the actual name is in group 1
                    if entity == "NAME_LABELED" and m.lastindex:
                        start = m.start(1)
                        end = m.end(1)
                        matched_text = m.group(1)
                    else:
                        start = m.start()
                        end = m.end()
                        matched_text = m.group()

                    matches.append(
                        PHIMatch(
                            entity_type=entity,
                            start=start,
                            end=end,
                            text=matched_text,
                            confidence=0.95,
                            redacted_text="[REDACTED]",
                        )
                    )
        # Deduplicate overlapping matches (keep highest confidence)
        return _deduplicate_matches(matches)


class PresidioPHIEngine:
    """
    Presidio-based PHI detection engine.

    Install with: pip install medguard[nlp]
    After install: python -m spacy download en_core_web_lg
    """

    def __init__(self) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore[import]

            self._analyzer = AnalyzerEngine()
        except ImportError as e:
            raise ImportError(
                "presidio-analyzer is not installed. "
                "Install with: pip install 'medguard[nlp]'\n"
                "Then run: python -m spacy download en_core_web_lg"
            ) from e

    def analyze(self, text: str, entities: list[str]) -> list[PHIMatch]:
        # Map medguard entity names to Presidio entity names
        presidio_entities = _map_to_presidio_entities(entities)
        results = self._analyzer.analyze(
            text=text,
            entities=presidio_entities,
            language="en",
        )
        return [
            PHIMatch(
                entity_type=r.entity_type,
                start=r.start,
                end=r.end,
                text=text[r.start : r.end],
                confidence=r.score,
                redacted_text="[REDACTED]",
            )
            for r in results
        ]


def _map_to_presidio_entities(entities: list[str]) -> list[str]:
    mapping = {
        "SSN": "US_SSN",
        "PHONE": "PHONE_NUMBER",
        "EMAIL": "EMAIL_ADDRESS",
        "DOB": "DATE_TIME",
        "MRN": "MEDICAL_LICENSE",
        "NPI": "US_NPI",
        "ZIP": "US_ZIPCODE",
        "ADDRESS": "LOCATION",
        "NAME_LABELED": "PERSON",
    }
    return [mapping.get(e, e) for e in entities]


def _deduplicate_matches(matches: list[PHIMatch]) -> list[PHIMatch]:
    """Remove overlapping matches, keeping the one with higher confidence."""
    if not matches:
        return matches
    sorted_matches = sorted(matches, key=lambda m: (m.start, -m.confidence))
    deduped: list[PHIMatch] = []
    last_end = -1
    for m in sorted_matches:
        if m.start >= last_end:
            deduped.append(m)
            last_end = m.end
    return deduped


class PHIDetector:
    """
    Facade that selects and delegates to the appropriate PHI engine.

    Engine selection order (first available wins):
      1. Explicit engine in config
      2. 'presidio' if presidio-analyzer is installed
      3. 'regex' (always available)
    """

    def __init__(self, config: PHIConfig) -> None:
        self.config = config
        self._engine = self._select_engine()

    def _select_engine(self) -> RegexPHIEngine | PresidioPHIEngine:
        if self.config.engine == "presidio":
            return PresidioPHIEngine()
        return RegexPHIEngine()

    def detect(self, text: str) -> PHIResult:
        matches = self._engine.analyze(text, self.config.entities)
        processed = text
        if self.config.mode == "redact" and matches:
            processed = _apply_redactions(text, matches, self.config.redaction_placeholder)
        return PHIResult(
            original=text,
            processed=processed,
            matches=matches,
            phi_detected=bool(matches),
            engine_used=type(self._engine).__name__,
        )

    def redact(self, text: str) -> str:
        """Convenience wrapper — returns redacted text directly."""
        result = self.detect(text)
        return result.processed


def _apply_redactions(text: str, matches: list[PHIMatch], placeholder: str) -> str:
    """Replace all matched spans with the placeholder, processing right-to-left
    so earlier offsets remain valid after each replacement."""
    sorted_matches = sorted(matches, key=lambda m: m.start, reverse=True)
    chars = list(text)
    for m in sorted_matches:
        chars[m.start : m.end] = list(placeholder)
    return "".join(chars)
