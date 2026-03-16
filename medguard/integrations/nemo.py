"""
NeMo Guardrails integration.

Registers medguard's pipeline stages as NeMo-callable Python actions
and provides a bundled Colang config for medical domain flows.

Usage:
    from nemoguardrails import RailsConfig, LLMRails
    from medguard import MedGuard

    mg = MedGuard()
    actions = mg.as_nemo_actions()

    config = RailsConfig.from_content(
        colang_content=MEDICAL_COLANG_CONFIG,
        yaml_content=NEMO_YAML_CONFIG,
    )
    rails = LLMRails(config)
    for name, action in actions.items():
        rails.register_action(action, name=name)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medguard.core import MedGuard


MEDICAL_COLANG_CONFIG = '''
define user ask medical question
  "what medication should I take"
  "is this drug safe"
  "what are the side effects of"
  "I have symptoms of"
  "what is the treatment for"
  "my doctor prescribed"
  "what is the dosage for"

define flow check medical safety
  $phi_result = execute check_phi(text=$user_message)
  if $phi_result.phi_detected
    bot say "I noticed your message may contain personal health information. It has been redacted for privacy before processing."

  $scope_result = execute check_scope(text=$user_message)
  if not $scope_result.in_scope
    bot say $scope_result.reason
    stop

  $drug_result = execute check_drug_safety(text=$user_message)
  if $drug_result.blocked
    bot say "I cannot provide guidance on this medication combination due to serious drug interaction risks. Please consult a healthcare professional or pharmacist immediately."
    stop

  if $drug_result.warnings
    $warning_text = $drug_result.warnings[0]
    bot say "Important drug safety notice: " + $warning_text

define flow check output safety
  $hall_result = execute check_hallucination(text=$bot_message)
  if $hall_result.blocked
    bot say "I need to correct something I said. The previous response may contain inaccurate medical information. Please consult a healthcare professional for accurate guidance."

define bot medical disclaimer
  "Please note: This information is for educational purposes only and does not constitute medical advice. Always consult a qualified healthcare professional for personal medical decisions."
'''

NEMO_YAML_CONFIG = '''
models:
  - type: main
    engine: openai
    model: gpt-4o-mini

instructions:
  - type: general
    content: |
      You are a helpful medical information assistant. Provide accurate,
      evidence-based information. Always recommend consulting a healthcare
      professional for personal medical decisions.
'''


def build_nemo_actions(mg: MedGuard) -> dict:
    """
    Build a dict of NeMo-compatible async action handlers from a MedGuard instance.

    Register each action with:
        rails.register_action(action, name=name)
    """
    actions = {}

    async def check_phi(context: dict, **kwargs) -> dict:
        text = context.get("user_message", "")
        if mg.phi_detector is None:
            return {"phi_detected": False, "matches": []}
        result = mg.phi_detector.detect(text)
        return {
            "phi_detected": result.phi_detected,
            "processed_text": result.processed,
            "matches": [
                {"entity_type": m.entity_type, "text": m.text}
                for m in result.matches
            ],
        }

    async def check_scope(context: dict, **kwargs) -> dict:
        text = context.get("user_message", "")
        if mg.scope_enforcer is None:
            return {"in_scope": True, "category": "unknown", "reason": None}
        result = mg.scope_enforcer.check(text)
        return result.to_dict()

    async def check_drug_safety(context: dict, **kwargs) -> dict:
        text = context.get("user_message", "")
        if mg.drug_checker is None:
            return {"blocked": False, "interactions": [], "warnings": []}
        result = await mg.drug_checker.check(text)
        return {
            "blocked": result.blocked,
            "interactions": [
                {
                    "drug_a": i.drug_a,
                    "drug_b": i.drug_b,
                    "severity": i.severity.value,
                    "description": i.description,
                }
                for i in result.interactions
            ],
            "warnings": result.warnings,
            "highest_severity": result.highest_severity.value,
        }

    async def check_hallucination(context: dict, **kwargs) -> dict:
        text = context.get("bot_message", "")
        if mg.hallucination_detector is None:
            return {"blocked": False, "flags": [], "hallucination_score": 0.0}
        result = await mg.hallucination_detector.check(text)
        return {
            "blocked": result.blocked,
            "flags": [
                {
                    "type": f.type.value,
                    "text": f.text,
                    "explanation": f.explanation,
                }
                for f in result.flags
            ],
            "hallucination_score": result.hallucination_score,
            "annotated_text": result.annotated_text,
        }

    actions["check_phi"] = check_phi
    actions["check_scope"] = check_scope
    actions["check_drug_safety"] = check_drug_safety
    actions["check_hallucination"] = check_hallucination

    return actions
