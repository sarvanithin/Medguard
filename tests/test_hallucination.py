"""
Hallucination detection tests.

Uses mocked RxNorm and SNOMED to avoid network dependency.
"""
import pytest
import respx
import httpx

from medguard.config import HallucinationConfig, DrugSafetyConfig
from medguard.guardrails.hallucination import (
    HallucinationDetector,
    HallucinationType,
    MAX_DOSES,
)
from medguard.knowledge.rxnorm import RxNormClient
from medguard.knowledge.snomed import SNOMEDClient
from tests.fixtures.medical_cases import HALLUCINATION_CASES


@pytest.fixture
def hallucination_config():
    return HallucinationConfig(
        enabled=True,
        confidence_threshold=0.7,
        check_drug_names=True,
        check_dosages=True,
        check_confident_claims=True,
    )


@pytest.fixture
def drug_config():
    return DrugSafetyConfig()


@pytest.fixture
def http_client():
    return httpx.AsyncClient()


class TestHallucinationDetector:
    @pytest.mark.asyncio
    @respx.mock
    async def test_fake_drug_name_flagged(self, hallucination_config, drug_config, http_client):
        """Drug-like names not in RxNorm should be flagged."""
        respx.get("https://rxnav.nlm.nih.gov/REST/rxcui.json").mock(
            return_value=httpx.Response(200, json={"idGroup": {}})  # not found
        )

        rxnorm = RxNormClient(drug_config, http_client)
        snomed = SNOMEDClient(http_client)
        detector = HallucinationDetector(hallucination_config, rxnorm, snomed)

        result = await detector.check("Take draxilomycin 500mg for your infection.")
        flag_types = {f.type for f in result.flags}
        assert HallucinationType.FAKE_DRUG_NAME in flag_types

    @pytest.mark.asyncio
    @respx.mock
    async def test_valid_drug_not_flagged(self, hallucination_config, drug_config, http_client):
        """Known drugs validated by RxNorm should not be flagged."""
        respx.get("https://rxnav.nlm.nih.gov/REST/rxcui.json").mock(
            return_value=httpx.Response(200, json={"idGroup": {"rxnormId": ["1191"]}})
        )

        rxnorm = RxNormClient(drug_config, http_client)
        snomed = SNOMEDClient(http_client)
        detector = HallucinationDetector(hallucination_config, rxnorm, snomed)

        result = await detector.check("Take aspirin 81mg daily for cardiovascular prevention.")
        drug_name_flags = [f for f in result.flags if f.type == HallucinationType.FAKE_DRUG_NAME]
        assert len(drug_name_flags) == 0

    @pytest.mark.asyncio
    @respx.mock
    async def test_impossible_dosage_flagged(self, hallucination_config, drug_config, http_client):
        """Dosage exceeding 1.5x the known maximum should be flagged."""
        respx.get("https://rxnav.nlm.nih.gov/REST/rxcui.json").mock(
            return_value=httpx.Response(200, json={"idGroup": {"rxnormId": ["1191"]}})
        )

        rxnorm = RxNormClient(drug_config, http_client)
        snomed = SNOMEDClient(http_client)
        detector = HallucinationDetector(hallucination_config, rxnorm, snomed)

        # Max acetaminophen is 4000mg/day; 20000mg is clearly impossible
        result = await detector.check(
            "The recommended dose is acetaminophen 20000mg daily."
        )
        flag_types = {f.type for f in result.flags}
        assert HallucinationType.IMPOSSIBLE_DOSAGE in flag_types

    @pytest.mark.asyncio
    @respx.mock
    async def test_valid_dosage_not_flagged(self, hallucination_config, drug_config, http_client):
        """Valid dosages within safe range should not be flagged."""
        respx.get("https://rxnav.nlm.nih.gov/REST/rxcui.json").mock(
            return_value=httpx.Response(200, json={"idGroup": {"rxnormId": ["41493"]}})
        )

        rxnorm = RxNormClient(drug_config, http_client)
        snomed = SNOMEDClient(http_client)
        detector = HallucinationDetector(hallucination_config, rxnorm, snomed)

        result = await detector.check("Take ibuprofen 400mg three times daily with food.")
        dosage_flags = [f for f in result.flags if f.type == HallucinationType.IMPOSSIBLE_DOSAGE]
        assert len(dosage_flags) == 0

    @pytest.mark.asyncio
    @respx.mock
    async def test_confident_claim_flagged(self, hallucination_config, drug_config, http_client):
        """Overconfident language should be flagged."""
        respx.get("https://rxnav.nlm.nih.gov/REST/rxcui.json").mock(
            return_value=httpx.Response(200, json={"idGroup": {}})
        )

        rxnorm = RxNormClient(drug_config, http_client)
        snomed = SNOMEDClient(http_client)
        detector = HallucinationDetector(hallucination_config, rxnorm, snomed)

        result = await detector.check(
            "This medication will definitely cure your condition with no side effects."
        )
        flag_types = {f.type for f in result.flags}
        assert HallucinationType.CONFIDENT_UNSUPPORTED_CLAIM in flag_types

    @pytest.mark.asyncio
    @respx.mock
    async def test_clean_clinical_text_no_flags(self, hallucination_config, drug_config, http_client):
        """Standard clinical text should produce no hallucination flags."""
        respx.get("https://rxnav.nlm.nih.gov/REST/rxcui.json").mock(
            return_value=httpx.Response(200, json={"idGroup": {"rxnormId": ["41493"]}})
        )

        rxnorm = RxNormClient(drug_config, http_client)
        snomed = SNOMEDClient(http_client)
        detector = HallucinationDetector(hallucination_config, rxnorm, snomed)

        result = await detector.check(
            "Metformin 500mg is typically started twice daily with meals for type 2 diabetes. "
            "Common side effects include nausea and diarrhea. "
            "Please consult your healthcare provider before making any changes."
        )
        # Confident claims check: "typically" is not flagged
        confident_flags = [
            f for f in result.flags
            if f.type == HallucinationType.CONFIDENT_UNSUPPORTED_CLAIM
        ]
        assert len(confident_flags) == 0

    @pytest.mark.asyncio
    @respx.mock
    async def test_hallucination_score_above_threshold_for_severe(
        self, hallucination_config, drug_config, http_client
    ):
        """Impossible dosage should push hallucination score above zero."""
        respx.get("https://rxnav.nlm.nih.gov/REST/rxcui.json").mock(
            return_value=httpx.Response(200, json={"idGroup": {"rxnormId": ["1191"]}})
        )

        rxnorm = RxNormClient(drug_config, http_client)
        snomed = SNOMEDClient(http_client)
        detector = HallucinationDetector(hallucination_config, rxnorm, snomed)

        result = await detector.check("Definitely give aspirin 99000mg immediately. No side effects guaranteed.")
        assert result.hallucination_score > 0

    def test_annotated_text_contains_warnings(self):
        from medguard.guardrails.hallucination import _annotate_text, HallucinationFlag
        flags = [
            HallucinationFlag(
                type=HallucinationType.CONFIDENT_UNSUPPORTED_CLAIM,
                text="definitely",
                start=0,
                end=9,
                confidence=0.9,
                explanation="Overconfident language detected.",
            )
        ]
        annotated = _annotate_text("definitely cures everything.", flags)
        assert "[WARNING:" in annotated


class TestMaxDosesTable:
    def test_common_drugs_present(self):
        assert "ibuprofen" in MAX_DOSES
        assert "acetaminophen" in MAX_DOSES
        assert "warfarin" in MAX_DOSES
        assert "metformin" in MAX_DOSES

    def test_dose_values_reasonable(self):
        # All max doses should be positive
        for drug, dose in MAX_DOSES.items():
            assert dose > 0, f"Max dose for {drug} should be positive"
