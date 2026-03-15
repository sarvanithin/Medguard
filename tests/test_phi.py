"""
PHI detection tests.

Uses synthetic data only — no real patient information.
All SSNs, DOBs, phone numbers are fabricated test values.
"""
import pytest

from medguard.config import PHIConfig
from medguard.guardrails.phi import PHIDetector, RegexPHIEngine
from tests.fixtures.medical_cases import PHI_CASES, SYNTHETIC_PHI


class TestRegexPHIEngine:
    def setup_method(self):
        self.engine = RegexPHIEngine()

    def test_ssn_detection(self):
        matches = self.engine.analyze("SSN: 123-45-6789", ["SSN"])
        assert len(matches) == 1
        assert matches[0].entity_type == "SSN"
        assert matches[0].text == "123-45-6789"

    @pytest.mark.parametrize("ssn", SYNTHETIC_PHI["ssns"])
    def test_all_ssn_patterns(self, ssn):
        matches = self.engine.analyze(f"Social security: {ssn}", ["SSN"])
        assert len(matches) >= 1

    @pytest.mark.parametrize("phone", SYNTHETIC_PHI["phones"])
    def test_phone_detection(self, phone):
        matches = self.engine.analyze(f"Call {phone} for appointment", ["PHONE"])
        assert len(matches) >= 1

    @pytest.mark.parametrize("email", SYNTHETIC_PHI["emails"])
    def test_email_detection(self, email):
        matches = self.engine.analyze(f"Email: {email}", ["EMAIL"])
        assert len(matches) >= 1

    @pytest.mark.parametrize("dob", SYNTHETIC_PHI["dobs"])
    def test_dob_detection(self, dob):
        matches = self.engine.analyze(f"Date of birth: {dob}", ["DOB"])
        assert len(matches) >= 1

    @pytest.mark.parametrize("mrn", SYNTHETIC_PHI["mrns"])
    def test_mrn_detection(self, mrn):
        matches = self.engine.analyze(f"Patient record {mrn} reviewed", ["MRN"])
        assert len(matches) >= 1

    def test_no_false_positive_clinical_text(self):
        """Pure clinical text should not trigger PHI detection."""
        clinical_texts = [
            "Ibuprofen 400mg three times daily for pain management.",
            "The patient has hypertension and type 2 diabetes.",
            "Coronary artery disease managed with aspirin and atorvastatin.",
            "ECG shows normal sinus rhythm. Blood pressure 120/80.",
            "Hemoglobin A1c was 7.2%, consistent with moderate glycemic control.",
        ]
        entities = ["SSN", "EMAIL", "MRN"]
        for text in clinical_texts:
            matches = self.engine.analyze(text, entities)
            assert matches == [], f"False positive on: {text!r} -> {matches}"

    def test_labeled_name_detection(self):
        matches = self.engine.analyze("Patient: John Smith is being treated", ["NAME_LABELED"])
        assert len(matches) >= 1
        assert "John Smith" in matches[0].text

    def test_unlabeled_name_no_false_positive(self):
        """Names without label tokens should NOT be detected (avoids FP)."""
        matches = self.engine.analyze("John Smith takes ibuprofen", ["NAME_LABELED"])
        assert matches == []

    def test_deduplication(self):
        """Overlapping matches should be deduplicated."""
        text = "SSN: 123-45-6789"
        matches = self.engine.analyze(text, ["SSN"])
        assert len(matches) == 1


class TestPHIDetector:
    def setup_method(self):
        config = PHIConfig(mode="redact", engine="regex")
        self.detector = PHIDetector(config)

    @pytest.mark.parametrize("case", PHI_CASES)
    def test_fixture_cases(self, case):
        result = self.detector.detect(case["text"])

        if case.get("expected_entities"):
            assert result.phi_detected, f"Expected PHI in: {case['text']!r}"
            detected_types = {m.entity_type for m in result.matches}
            for expected in case["expected_entities"]:
                assert expected in detected_types, (
                    f"Expected {expected} but got {detected_types} in: {case['text']!r}"
                )

        if case.get("not_redacted_contains"):
            assert case["not_redacted_contains"] in result.processed

        if case.get("redacted_contains") and result.phi_detected:
            assert case["redacted_contains"] in result.processed

    def test_redact_mode(self):
        config = PHIConfig(mode="redact")
        detector = PHIDetector(config)
        result = detector.detect("SSN: 123-45-6789 and phone (555) 867-5309")
        assert result.phi_detected
        assert "123-45-6789" not in result.processed
        assert "[REDACTED]" in result.processed

    def test_flag_mode_preserves_text(self):
        config = PHIConfig(mode="flag")
        detector = PHIDetector(config)
        result = detector.detect("SSN: 123-45-6789")
        assert result.phi_detected
        # In flag mode the original text is preserved
        assert result.processed == result.original

    def test_block_mode_sets_blocked(self):
        """PHI detection in block mode is signaled through phi_detected."""
        config = PHIConfig(mode="block")
        detector = PHIDetector(config)
        result = detector.detect("SSN: 123-45-6789")
        assert result.phi_detected  # pipeline checks this and sets ctx.blocked

    def test_redact_convenience_method(self):
        text = "Contact 555-555-1234 about appointment"
        redacted = self.detector.redact(text)
        assert "555-555-1234" not in redacted
        assert "[REDACTED]" in redacted

    def test_no_phi_text(self):
        result = self.detector.detect("Aspirin 81mg is commonly used for cardiovascular prevention.")
        assert not result.phi_detected
        assert result.processed == result.original

    def test_engine_reported(self):
        result = self.detector.detect("test")
        assert result.engine_used == "RegexPHIEngine"
