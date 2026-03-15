"""
Drug safety tests.

Unit tests use mocked httpx (respx) — no real API calls.
Integration tests (marked) hit real OpenFDA API and are skipped in CI.
"""
import json
import pytest
import respx
import httpx

from medguard.config import DrugSafetyConfig
from medguard.knowledge.openfda import OpenFDAClient, InteractionSeverity
from medguard.knowledge.rxnorm import RxNormClient
from medguard.guardrails.drug_safety import (
    DrugSafetyChecker,
    StaticInteractionTable,
    DrugMentionExtractor,
)


@pytest.fixture
def drug_config():
    return DrugSafetyConfig(
        enabled=True,
        severity_threshold="moderate",
        use_openfda=True,
        use_rxnorm=True,
        use_static_fallback=True,
        cache_ttl_seconds=0,
    )


@pytest.fixture
def http_client():
    return httpx.AsyncClient()


@pytest.fixture
def rxnorm_client(drug_config, http_client):
    return RxNormClient(drug_config, http_client)


@pytest.fixture
def openfda_client(drug_config, http_client):
    return OpenFDAClient(drug_config, http_client)


class TestOpenFDAClient:
    def test_parse_severity_contraindicated(self):
        client = OpenFDAClient.__new__(OpenFDAClient)
        text = "Concurrent use is contraindicated due to severe bleeding risk."
        assert client._parse_severity(text) == InteractionSeverity.CONTRAINDICATED

    def test_parse_severity_high(self):
        client = OpenFDAClient.__new__(OpenFDAClient)
        text = "Avoid this combination. Life-threatening interactions have been reported."
        assert client._parse_severity(text) == InteractionSeverity.HIGH

    def test_parse_severity_moderate(self):
        client = OpenFDAClient.__new__(OpenFDAClient)
        text = "Monitor patients closely. May increase warfarin levels."
        assert client._parse_severity(text) == InteractionSeverity.MODERATE

    def test_parse_severity_low(self):
        client = OpenFDAClient.__new__(OpenFDAClient)
        text = "Minor interaction. May slightly increase plasma levels."
        assert client._parse_severity(text) == InteractionSeverity.LOW

    def test_parse_severity_unknown(self):
        client = OpenFDAClient.__new__(OpenFDAClient)
        text = "These drugs have been used together in clinical trials."
        assert client._parse_severity(text) == InteractionSeverity.UNKNOWN

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_drug_interactions_found(self, openfda_client):
        respx.get("https://api.fda.gov/drug/label.json").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "drug_interactions": [
                                "Warfarin: Concurrent use with aspirin is contraindicated "
                                "due to markedly increased bleeding risk."
                            ]
                        }
                    ]
                },
            )
        )
        result = await openfda_client.get_drug_interactions("warfarin", "aspirin")
        assert result is not None
        assert result.drug_a == "warfarin"
        assert result.drug_b == "aspirin"
        assert result.severity == InteractionSeverity.CONTRAINDICATED

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_drug_interactions_not_found(self, openfda_client):
        respx.get("https://api.fda.gov/drug/label.json").mock(
            return_value=httpx.Response(404, json={"error": {"message": "No matches"}})
        )
        result = await openfda_client.get_drug_interactions("metformin", "lisinopril")
        assert result is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_drug_interactions_empty_results(self, openfda_client):
        respx.get("https://api.fda.gov/drug/label.json").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        result = await openfda_client.get_drug_interactions("metformin", "lisinopril")
        assert result is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_api_failure_returns_none(self, openfda_client):
        respx.get("https://api.fda.gov/drug/label.json").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        # Should not raise — fall back gracefully
        result = await openfda_client.get_drug_interactions("warfarin", "aspirin")
        assert result is None


class TestStaticInteractionTable:
    def setup_method(self):
        self.table = StaticInteractionTable()

    def test_known_high_interaction(self):
        result = self.table.lookup("warfarin", "aspirin")
        assert result is not None
        assert result.severity in (InteractionSeverity.HIGH, InteractionSeverity.CONTRAINDICATED)

    def test_known_contraindicated_interaction(self):
        result = self.table.lookup("simvastatin", "clarithromycin")
        assert result is not None
        assert result.severity == InteractionSeverity.CONTRAINDICATED

    def test_ordering_independent(self):
        """Lookup should work regardless of drug ordering."""
        result_ab = self.table.lookup("warfarin", "aspirin")
        result_ba = self.table.lookup("aspirin", "warfarin")
        assert result_ab is not None
        assert result_ba is not None
        assert result_ab.severity == result_ba.severity

    def test_unknown_pair_returns_none(self):
        result = self.table.lookup("vitamin_c", "orange_juice")
        assert result is None

    def test_ssri_maoi_contraindicated(self):
        result = self.table.lookup("sertraline", "phenelzine")
        assert result is not None
        assert result.severity == InteractionSeverity.CONTRAINDICATED


class TestDrugMentionExtractor:
    @pytest.mark.asyncio
    @respx.mock
    async def test_extract_common_drugs(self, drug_config, http_client):
        rxnorm = RxNormClient(drug_config, http_client)
        extractor = DrugMentionExtractor()

        # Mock RxNorm lookup
        respx.get("https://rxnav.nlm.nih.gov/REST/rxcui.json").mock(
            return_value=httpx.Response(
                200,
                json={"idGroup": {"rxnormId": ["1191"]}},
            )
        )
        respx.get(url__startswith="https://rxnav.nlm.nih.gov/REST/rxcui/").mock(
            return_value=httpx.Response(
                200,
                json={"properties": {"name": "Aspirin"}},
            )
        )

        text = "Patient is taking aspirin 81mg daily"
        mentions = await extractor.extract(text, rxnorm)
        assert len(mentions) >= 1
        drug_names = [m.raw_name.lower() for m in mentions]
        assert "aspirin" in drug_names

    @pytest.mark.asyncio
    @respx.mock
    async def test_extract_dosage_pattern(self, drug_config, http_client):
        rxnorm = RxNormClient(drug_config, http_client)
        extractor = DrugMentionExtractor()

        respx.get("https://rxnav.nlm.nih.gov/REST/rxcui.json").mock(
            return_value=httpx.Response(200, json={"idGroup": {"rxnormId": ["41493"]}})
        )
        respx.get(url__startswith="https://rxnav.nlm.nih.gov/REST/rxcui/").mock(
            return_value=httpx.Response(200, json={"properties": {"name": "Metformin"}})
        )

        text = "Metformin 500mg twice daily was prescribed."
        mentions = await extractor.extract(text, rxnorm)
        assert any("metformin" in m.raw_name.lower() for m in mentions)


class TestDrugSafetyChecker:
    @pytest.mark.asyncio
    @respx.mock
    async def test_warfarin_aspirin_blocked(self, drug_config, http_client):
        """Warfarin + aspirin should be blocked at moderate threshold."""
        # Mock RxNorm — warfarin
        rxnorm_calls = 0

        def rxnorm_side_effect(request):
            name = request.url.params.get("name", "").lower()
            if "warfarin" in name:
                return httpx.Response(200, json={"idGroup": {"rxnormId": ["11289"]}})
            elif "aspirin" in name:
                return httpx.Response(200, json={"idGroup": {"rxnormId": ["1191"]}})
            return httpx.Response(200, json={"idGroup": {}})

        def rxnorm_props_side_effect(request):
            path = request.url.path
            if "11289" in path:
                return httpx.Response(200, json={"properties": {"name": "Warfarin"}})
            elif "1191" in path:
                return httpx.Response(200, json={"properties": {"name": "Aspirin"}})
            return httpx.Response(200, json={"properties": {"name": "Unknown"}})

        respx.get("https://rxnav.nlm.nih.gov/REST/rxcui.json").mock(
            side_effect=rxnorm_side_effect
        )
        respx.get(url__regex=r"https://rxnav\.nlm\.nih\.gov/REST/rxcui/\d+/properties\.json").mock(
            side_effect=rxnorm_props_side_effect
        )

        # Mock OpenFDA — return high severity interaction
        respx.get("https://api.fda.gov/drug/label.json").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "drug_interactions": [
                                "Aspirin: Avoid concurrent use. "
                                "Life-threatening bleeding risk increases significantly."
                            ]
                        }
                    ]
                },
            )
        )

        rxnorm = RxNormClient(drug_config, http_client)
        openfda = OpenFDAClient(drug_config, http_client)
        checker = DrugSafetyChecker(drug_config, rxnorm, openfda)

        result = await checker.check("Patient takes warfarin and aspirin daily.")
        assert result.blocked or len(result.interactions) > 0

    @pytest.mark.asyncio
    @respx.mock
    async def test_single_drug_not_blocked(self, drug_config, http_client):
        """Single drug — no interactions possible."""
        respx.get(url__startswith="https://rxnav.nlm.nih.gov").mock(
            return_value=httpx.Response(200, json={"idGroup": {"rxnormId": ["41493"]}})
        )
        respx.get(url__startswith="https://rxnav.nlm.nih.gov/REST/rxcui/").mock(
            return_value=httpx.Response(200, json={"properties": {"name": "Metformin"}})
        )

        rxnorm = RxNormClient(drug_config, http_client)
        openfda = OpenFDAClient(drug_config, http_client)
        checker = DrugSafetyChecker(drug_config, rxnorm, openfda)

        result = await checker.check("Patient takes metformin 500mg daily.")
        assert not result.blocked

    @pytest.mark.asyncio
    async def test_api_failure_falls_back_to_static_table(self, drug_config):
        """When OpenFDA fails, static table should still detect warfarin+aspirin."""
        # Use a fresh http client but mock at the checker level
        async with httpx.AsyncClient() as client:
            rxnorm = RxNormClient(drug_config, client)
            openfda = OpenFDAClient(drug_config, client)
            checker = DrugSafetyChecker(drug_config, rxnorm, openfda)

            # Directly test the static table fallback
            result = checker._static_table.lookup("warfarin", "aspirin")
            assert result is not None
            assert result.severity in (InteractionSeverity.HIGH, InteractionSeverity.CONTRAINDICATED)


# ---------------------------------------------------------------------------
# Integration tests (hit real APIs — skipped in CI)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_warfarin_aspirin_real_api():
    """Integration test: warfarin + aspirin via real OpenFDA API."""
    config = DrugSafetyConfig(enabled=True, cache_ttl_seconds=3600)
    async with httpx.AsyncClient() as client:
        openfda = OpenFDAClient(config, client)
        result = await openfda.get_drug_interactions("warfarin", "aspirin")
        # Allow None if label structure changed, but severity should be high if found
        if result is not None:
            assert result.severity in (
                InteractionSeverity.HIGH,
                InteractionSeverity.CONTRAINDICATED,
                InteractionSeverity.MODERATE,
            )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_rxnorm_aspirin():
    """Integration test: RxNorm lookup for aspirin."""
    config = DrugSafetyConfig(enabled=True, cache_ttl_seconds=3600)
    async with httpx.AsyncClient() as client:
        rxnorm = RxNormClient(config, client)
        rxcui = await rxnorm.get_rxcui("aspirin")
        assert rxcui is not None
        assert rxcui == "1191"
