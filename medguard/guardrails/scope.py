"""
Clinical scope enforcement.

Determines whether a user query is within the clinical scope of a medical
assistant. Uses keyword classification as the primary method (fast, zero-API).

Phase 2 roadmap: Add a ScopeClassifierProtocol-compatible ML classifier
using distilbert-base-uncased fine-tuned on MedMCQA + HealthSearchQA.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from medguard.config import ScopeConfig


class ScopeCategory(str, Enum):
    CLINICAL_DIAGNOSIS = "clinical_diagnosis"
    CLINICAL_TREATMENT = "clinical_treatment"
    CLINICAL_MEDICATION = "clinical_medication"
    CLINICAL_ANATOMY = "clinical_anatomy"
    CLINICAL_GENERAL = "clinical_general"
    OUT_OF_SCOPE_LEGAL = "out_of_scope_legal"
    OUT_OF_SCOPE_FINANCIAL = "out_of_scope_financial"
    AMBIGUOUS = "ambiguous"


class ScopeResult:
    def __init__(
        self,
        category: ScopeCategory,
        in_scope: bool,
        confidence: float,
        action_taken: Literal["pass", "warn", "block"],
        reason: str | None = None,
    ) -> None:
        self.category = category
        self.in_scope = in_scope
        self.confidence = confidence
        self.action_taken = action_taken
        self.reason = reason

    def to_dict(self) -> dict:
        return {
            "category": self.category.value,
            "in_scope": self.in_scope,
            "confidence": self.confidence,
            "action_taken": self.action_taken,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------

_CLINICAL_DIAGNOSIS_KEYWORDS: frozenset[str] = frozenset([
    "symptom", "symptoms", "diagnosis", "diagnose", "condition", "disease",
    "chest pain", "shortness of breath", "i have", "i am having", "i feel",
    "pain", "ache", "hurt", "hurts", "swelling", "bleeding", "nausea",
    "fever", "cough", "fatigue", "dizzy", "dizziness", "rash", "headache",
    "disorder", "syndrome", "pathology", "differential", "prognosis",
    "presenting", "complaint", "chief complaint", "history of", "signs of",
    "signs and symptoms", "test result", "lab result", "biopsy", "blood test",
    "imaging", "mri", "ct scan", "x-ray", "ultrasound", "echocardiogram",
    "EKG", "ECG", "colonoscopy", "endoscopy", "do i have", "could this be",
    "what is wrong", "what causes",
])

_CLINICAL_TREATMENT_KEYWORDS: frozenset[str] = frozenset([
    "treatment", "treat", "therapy", "therapies", "procedure", "surgery",
    "operation", "intervention", "protocol", "guideline", "management",
    "care plan", "recovery", "rehabilitation", "physical therapy",
    "chemotherapy", "radiation", "immunotherapy", "transplant", "dialysis",
    "vaccine", "vaccination", "immunization", "preventive", "prevention",
    "how to treat", "how to manage", "what should i do",
])

_CLINICAL_MEDICATION_KEYWORDS: frozenset[str] = frozenset([
    "medication", "medicine", "drug", "drugs", "prescription", "dose", "dosage",
    "mg", "mcg", "milligram", "tablet", "capsule", "injection", "antibiotic",
    "antidepressant", "antihypertensive", "analgesic", "statin", "beta blocker",
    "ace inhibitor", "diuretic", "insulin", "steroid", "corticosteroid",
    "side effect", "side effects", "adverse effect", "contraindication",
    "drug interaction", "interaction", "allergy", "allergic reaction",
    "overdose", "withdrawal", "refill", "generic", "brand name",
])

_CLINICAL_ANATOMY_KEYWORDS: frozenset[str] = frozenset([
    "heart", "lung", "lungs", "liver", "kidney", "kidneys", "brain", "spine",
    "blood", "artery", "vein", "bone", "muscle", "nerve", "skin", "intestine",
    "colon", "stomach", "pancreas", "thyroid", "adrenal", "pituitary",
    "lymph node", "lymphatic", "immune", "hormone", "cell", "tissue", "organ",
    "anatomy", "physiology", "cardiovascular", "pulmonary", "renal", "hepatic",
    "neurological", "endocrine", "gastrointestinal", "musculoskeletal",
])

_CLINICAL_GENERAL_KEYWORDS: frozenset[str] = frozenset([
    "patient", "doctor", "physician", "nurse", "hospital", "clinic",
    "emergency", "urgent care", "specialist", "referral", "appointment",
    "health", "medical", "clinical", "healthcare", "wellness", "nutrition",
    "diet", "exercise", "mental health", "pregnancy", "pediatric", "geriatric",
    "chronic", "acute", "palliative", "hospice", "vital signs", "blood pressure",
    "heart rate", "temperature", "oxygen", "saturation",
])

_LEGAL_KEYWORDS: frozenset[str] = frozenset([
    "lawsuit", "sue", "suing", "malpractice", "negligence", "liability",
    "attorney", "lawyer", "legal action", "court", "settlement", "damages",
    "personal injury", "wrongful death", "statute of limitations",
    "medical malpractice", "standard of care violation",
])

_FINANCIAL_KEYWORDS: frozenset[str] = frozenset([
    "insurance coverage", "does my insurance", "will insurance cover",
    "my insurance", "insurance cover", "covered by insurance",
    "prior authorization", "preauthorization", "billing code", "CPT code",
    "ICD code", "deductible", "copay", "coinsurance", "out of pocket",
    "reimbursement", "cost of treatment",
    "how much does", "how much will", "financial assistance",
    "insurance plan", "health insurance", "insurance policy",
])


def _count_keyword_hits(text_lower: str, keywords: frozenset[str]) -> int:
    return sum(1 for kw in keywords if kw in text_lower)


class KeywordScopeClassifier:
    """
    Fast keyword-based scope classifier. Zero API calls, ~0ms latency.

    Returns (ScopeCategory, confidence) where confidence is based on
    the ratio of matched keywords to total checked.
    """

    def classify(self, text: str) -> tuple[ScopeCategory, float]:
        text_lower = text.lower()

        legal_hits = _count_keyword_hits(text_lower, _LEGAL_KEYWORDS)
        financial_hits = _count_keyword_hits(text_lower, _FINANCIAL_KEYWORDS)

        if legal_hits >= 1:
            confidence = min(0.7 + (legal_hits * 0.1), 0.99)
            return ScopeCategory.OUT_OF_SCOPE_LEGAL, confidence

        if financial_hits >= 1:
            confidence = min(0.7 + (financial_hits * 0.1), 0.99)
            return ScopeCategory.OUT_OF_SCOPE_FINANCIAL, confidence

        # Score clinical categories
        scores = {
            ScopeCategory.CLINICAL_MEDICATION: _count_keyword_hits(
                text_lower, _CLINICAL_MEDICATION_KEYWORDS
            ),
            ScopeCategory.CLINICAL_DIAGNOSIS: _count_keyword_hits(
                text_lower, _CLINICAL_DIAGNOSIS_KEYWORDS
            ),
            ScopeCategory.CLINICAL_TREATMENT: _count_keyword_hits(
                text_lower, _CLINICAL_TREATMENT_KEYWORDS
            ),
            ScopeCategory.CLINICAL_ANATOMY: _count_keyword_hits(
                text_lower, _CLINICAL_ANATOMY_KEYWORDS
            ),
            ScopeCategory.CLINICAL_GENERAL: _count_keyword_hits(
                text_lower, _CLINICAL_GENERAL_KEYWORDS
            ),
        }

        total_hits = sum(scores.values())
        if total_hits == 0:
            return ScopeCategory.AMBIGUOUS, 0.3

        best_category = max(scores, key=lambda c: scores[c])
        confidence = min(0.5 + (total_hits * 0.05), 0.95)
        return best_category, confidence


class ScopeEnforcer:
    """
    Applies scope enforcement rules based on classification results.
    """

    def __init__(self, config: ScopeConfig) -> None:
        self.config = config
        self._classifier = KeywordScopeClassifier()

    def check(self, text: str) -> ScopeResult:
        if not self.config.enabled:
            return ScopeResult(
                category=ScopeCategory.CLINICAL_GENERAL,
                in_scope=True,
                confidence=1.0,
                action_taken="pass",
            )

        category, confidence = self._classifier.classify(text)
        in_scope = category not in (
            ScopeCategory.OUT_OF_SCOPE_LEGAL,
            ScopeCategory.OUT_OF_SCOPE_FINANCIAL,
        )

        if not in_scope:
            action = self.config.action  # "warn" or "block"
            reason = _out_of_scope_reason(category)
        else:
            action = "pass"
            reason = None

        return ScopeResult(
            category=category,
            in_scope=in_scope,
            confidence=confidence,
            action_taken=action,
            reason=reason,
        )


def _out_of_scope_reason(category: ScopeCategory) -> str:
    if category == ScopeCategory.OUT_OF_SCOPE_LEGAL:
        return (
            "This question appears to involve legal matters. "
            "Please consult a qualified attorney for legal advice."
        )
    if category == ScopeCategory.OUT_OF_SCOPE_FINANCIAL:
        return (
            "This question appears to involve insurance or billing matters. "
            "Please contact your insurance provider or a billing specialist."
        )
    return "This question is outside the scope of medical assistance."
