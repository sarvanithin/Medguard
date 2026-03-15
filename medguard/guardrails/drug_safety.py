"""
Drug interaction and contraindication checking.

Pipeline:
  1. Extract drug mentions from text (regex + RxNorm validation)
  2. Normalize to canonical names via RxNorm
  3. For each drug pair, query OpenFDA interactions
  4. Fall back to static table on API failure
  5. Aggregate severity, apply threshold from config
"""
from __future__ import annotations

import asyncio
import csv
import re
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel

from medguard.config import DrugSafetyConfig
from medguard.knowledge.openfda import (
    ContraindicationResult,
    DrugInteraction,
    InteractionSeverity,
)

if TYPE_CHECKING:
    from medguard.knowledge.openfda import OpenFDAClient
    from medguard.knowledge.rxnorm import RxNormClient

log = structlog.get_logger(__name__)

_STATIC_TABLE_PATH = Path(__file__).parent.parent / "knowledge" / "data" / "drug_interactions.csv"

# Severity ordering for comparison
_SEVERITY_ORDER = {
    InteractionSeverity.UNKNOWN: 0,
    InteractionSeverity.LOW: 1,
    InteractionSeverity.MODERATE: 2,
    InteractionSeverity.HIGH: 3,
    InteractionSeverity.CONTRAINDICATED: 4,
}

# Dosage pattern: drug name followed by a dosage value
_DOSAGE_RE = re.compile(
    r"\b([A-Za-z][A-Za-z\s\-]{2,30}?)\s+"
    r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|mEq|units?|IU|ml|mL)\b",
    re.IGNORECASE,
)

# Standalone drug mention (from a curated common drug list)
_COMMON_DRUGS = frozenset([
    "warfarin", "aspirin", "ibuprofen", "naproxen", "metformin", "lisinopril",
    "atorvastatin", "simvastatin", "lovastatin", "metoprolol", "atenolol",
    "amlodipine", "losartan", "hydrochlorothiazide", "omeprazole", "pantoprazole",
    "sertraline", "fluoxetine", "paroxetine", "escitalopram", "citalopram",
    "amitriptyline", "nortriptyline", "phenelzine", "tranylcypromine",
    "clopidogrel", "apixaban", "rivaroxaban", "dabigatran", "heparin",
    "digoxin", "amiodarone", "quinidine", "lithium", "valproate", "carbamazepine",
    "phenytoin", "phenobarbital", "rifampin", "rifampicin", "clarithromycin",
    "erythromycin", "ciprofloxacin", "levofloxacin", "metronidazole",
    "fluconazole", "itraconazole", "ketoconazole", "voriconazole",
    "cyclosporine", "tacrolimus", "sirolimus", "methotrexate", "prednisone",
    "dexamethasone", "insulin", "glipizide", "glyburide", "acetaminophen",
    "codeine", "oxycodone", "morphine", "fentanyl", "tramadol", "naloxone",
    "sildenafil", "tadalafil", "finasteride", "tamsulosin", "allopurinol",
    "colchicine", "hydroxychloroquine", "azathioprine", "mercaptopurine",
])


class DrugMention(BaseModel):
    raw_name: str
    canonical_name: str | None = None
    rxcui: str | None = None
    confidence: float = 1.0


class DrugSafetyResult(BaseModel):
    drugs_found: list[DrugMention]
    interactions: list[DrugInteraction]
    contraindications: list[ContraindicationResult]
    highest_severity: InteractionSeverity
    blocked: bool
    warnings: list[str]


class DrugMentionExtractor:
    """
    Extracts drug names from free text.

    Strategy:
    1. Regex for dosage patterns ("aspirin 81mg")
    2. Exact match against curated common drug list
    3. RxNorm validation for any matches found
    """

    async def extract(
        self, text: str, rxnorm_client: RxNormClient
    ) -> list[DrugMention]:
        found: dict[str, DrugMention] = {}
        text_lower = text.lower()

        # Pass 1: dosage pattern extraction
        for match in _DOSAGE_RE.finditer(text):
            drug_name = match.group(1).strip().rstrip()
            if len(drug_name) < 3:
                continue
            found[drug_name.lower()] = DrugMention(raw_name=drug_name, confidence=0.9)

        # Pass 2: curated common drug list
        for drug in _COMMON_DRUGS:
            if drug in text_lower:
                if drug not in found:
                    found[drug] = DrugMention(raw_name=drug, confidence=0.95)

        if not found:
            return []

        # Pass 3: RxNorm validation and normalization
        async def validate(mention: DrugMention) -> DrugMention:
            try:
                rxcui = await rxnorm_client.get_rxcui(mention.raw_name)
                if rxcui:
                    canonical = await rxnorm_client.normalize_drug_name(rxcui)
                    return DrugMention(
                        raw_name=mention.raw_name,
                        canonical_name=canonical,
                        rxcui=rxcui,
                        confidence=mention.confidence,
                    )
            except Exception:
                pass
            return mention

        validated = await asyncio.gather(*[validate(m) for m in found.values()])
        return list(validated)


class StaticInteractionTable:
    """
    Offline fallback for drug interaction data.
    Loads medguard/knowledge/data/drug_interactions.csv.

    CSV columns: drug_a, drug_b, severity, description, source
    Lookup is by lowercase drug name (not RxCUI) for broader matching.
    """

    def __init__(self) -> None:
        self._table: dict[tuple[str, str], DrugInteraction] = {}
        self._load()

    def _load(self) -> None:
        if not _STATIC_TABLE_PATH.exists():
            log.warning("static_interaction_table_not_found", path=str(_STATIC_TABLE_PATH))
            return
        with _STATIC_TABLE_PATH.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                a = row["drug_a"].lower().strip()
                b = row["drug_b"].lower().strip()
                key = (min(a, b), max(a, b))  # canonical ordering
                try:
                    severity = InteractionSeverity(row["severity"].lower().strip())
                except ValueError:
                    severity = InteractionSeverity.UNKNOWN
                self._table[key] = DrugInteraction(
                    drug_a=row["drug_a"],
                    drug_b=row["drug_b"],
                    severity=severity,
                    description=row.get("description", ""),
                    source="static_table",
                    confidence=0.9,
                )

    def lookup(self, drug_a: str, drug_b: str) -> DrugInteraction | None:
        key = (min(drug_a.lower(), drug_b.lower()), max(drug_a.lower(), drug_b.lower()))
        return self._table.get(key)


class DrugSafetyChecker:
    def __init__(
        self,
        config: DrugSafetyConfig,
        rxnorm: RxNormClient,
        openfda: OpenFDAClient,
    ) -> None:
        self.config = config
        self._rxnorm = rxnorm
        self._openfda = openfda
        self._extractor = DrugMentionExtractor()
        self._static_table = StaticInteractionTable()

    async def check(self, text: str) -> DrugSafetyResult:
        drugs = await self._extractor.extract(text, self._rxnorm)

        if len(drugs) < 2:
            return DrugSafetyResult(
                drugs_found=drugs,
                interactions=[],
                contraindications=[],
                highest_severity=InteractionSeverity.UNKNOWN,
                blocked=False,
                warnings=[],
            )

        # Check all drug pairs concurrently
        pairs = [(drugs[i], drugs[j]) for i in range(len(drugs)) for j in range(i + 1, len(drugs))]
        interaction_results = await asyncio.gather(
            *[self._check_pair(a, b) for a, b in pairs]
        )

        interactions = [r for r in interaction_results if r is not None]

        # Get contraindications for all drugs
        all_contras: list[ContraindicationResult] = []
        if self.config.use_openfda:
            contra_results = await asyncio.gather(
                *[self._openfda.get_contraindications(d.canonical_name or d.raw_name) for d in drugs]
            )
            for contras in contra_results:
                all_contras.extend(contras)

        highest_severity = _compute_highest_severity(interactions)
        threshold = InteractionSeverity(self.config.severity_threshold)
        blocked = _severity_order(highest_severity) >= _severity_order(threshold)

        warnings = []
        for i in interactions:
            if _severity_order(i.severity) >= _severity_order(InteractionSeverity.MODERATE):
                warnings.append(
                    f"Drug interaction: {i.drug_a} + {i.drug_b} "
                    f"({i.severity.value}) — {i.description[:100]}"
                )

        return DrugSafetyResult(
            drugs_found=drugs,
            interactions=interactions,
            contraindications=all_contras,
            highest_severity=highest_severity,
            blocked=blocked,
            warnings=warnings,
        )

    async def _check_pair(
        self, drug_a: DrugMention, drug_b: DrugMention
    ) -> DrugInteraction | None:
        name_a = drug_a.canonical_name or drug_a.raw_name
        name_b = drug_b.canonical_name or drug_b.raw_name

        # Try OpenFDA first
        openfda_result = None
        if self.config.use_openfda:
            try:
                openfda_result = await self._openfda.get_drug_interactions(name_a, name_b)
            except Exception as exc:
                log.debug("openfda_pair_check_failed", error=str(exc))

        # Check static table (try canonical name, then raw name as fallback)
        static_result = None
        if self.config.use_static_fallback:
            static_result = self._static_table.lookup(name_a, name_b)
            if not static_result:
                # Try raw names (RxNorm may return full salt forms like "Sertraline Hydrochloride")
                static_result = self._static_table.lookup(drug_a.raw_name, drug_b.raw_name)

        # Return the result with highest severity
        if openfda_result and static_result:
            if _severity_order(static_result.severity) > _severity_order(openfda_result.severity):
                return static_result
            return openfda_result

        return openfda_result or static_result


def _compute_highest_severity(interactions: list[DrugInteraction]) -> InteractionSeverity:
    if not interactions:
        return InteractionSeverity.UNKNOWN
    return max(interactions, key=lambda i: _severity_order(i.severity)).severity


def _severity_order(severity: InteractionSeverity) -> int:
    return _SEVERITY_ORDER.get(severity, 0)
