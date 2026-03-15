"""
Clinical scope enforcement tests.
"""
import pytest

from medguard.config import ScopeConfig
from medguard.guardrails.scope import KeywordScopeClassifier, ScopeCategory, ScopeEnforcer
from tests.fixtures.medical_cases import SCOPE_CASES


class TestKeywordScopeClassifier:
    def setup_method(self):
        self.classifier = KeywordScopeClassifier()

    def test_medication_query_in_scope(self):
        category, confidence = self.classifier.classify("What medication should I take for my headache?")
        assert "clinical" in category.value
        assert confidence > 0.4

    def test_diagnosis_query_in_scope(self):
        category, confidence = self.classifier.classify("What are the symptoms of diabetes?")
        assert "clinical" in category.value

    def test_legal_query_out_of_scope(self):
        category, _ = self.classifier.classify("Can I sue my doctor for malpractice?")
        assert category == ScopeCategory.OUT_OF_SCOPE_LEGAL

    def test_financial_query_out_of_scope(self):
        category, _ = self.classifier.classify("Will my insurance cover this surgery?")
        assert category == ScopeCategory.OUT_OF_SCOPE_FINANCIAL

    def test_symptoms_in_scope(self):
        category, _ = self.classifier.classify("I have chest pain and shortness of breath")
        assert "clinical" in category.value

    def test_treatment_in_scope(self):
        category, _ = self.classifier.classify("What is the standard treatment for hypertension?")
        assert "clinical" in category.value

    def test_anatomy_in_scope(self):
        category, _ = self.classifier.classify("How does the heart pump blood to the lungs?")
        assert "clinical" in category.value


class TestScopeEnforcer:
    def setup_method(self):
        self.enforcer = ScopeEnforcer(ScopeConfig(enabled=True, action="warn"))

    @pytest.mark.parametrize("case", SCOPE_CASES)
    def test_fixture_cases(self, case):
        result = self.enforcer.check(case["text"])
        assert result.in_scope == case["expected_in_scope"], (
            f"Expected in_scope={case['expected_in_scope']} for: {case['text']!r}"
        )
        if "expected_category" in case:
            assert result.category.value == case["expected_category"]

    def test_warn_action_doesnt_block(self):
        enforcer = ScopeEnforcer(ScopeConfig(enabled=True, action="warn"))
        result = enforcer.check("Can I sue my doctor?")
        assert not result.in_scope
        assert result.action_taken == "warn"

    def test_block_action_blocks(self):
        enforcer = ScopeEnforcer(ScopeConfig(enabled=True, action="block"))
        result = enforcer.check("Can I sue my doctor?")
        assert not result.in_scope
        assert result.action_taken == "block"

    def test_disabled_enforcer_always_passes(self):
        enforcer = ScopeEnforcer(ScopeConfig(enabled=False))
        result = enforcer.check("Can I sue my doctor?")
        assert result.in_scope
        assert result.action_taken == "pass"

    def test_legal_reason_message(self):
        result = self.enforcer.check("I want to file a malpractice lawsuit")
        assert result.reason is not None
        assert "legal" in result.reason.lower() or "attorney" in result.reason.lower()

    def test_financial_reason_message(self):
        result = self.enforcer.check("Does my insurance cover this treatment?")
        assert result.reason is not None

    def test_clinical_query_passes(self):
        result = self.enforcer.check("What is the recommended dose of metformin for type 2 diabetes?")
        assert result.in_scope
        assert result.action_taken == "pass"
