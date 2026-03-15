"""
medguard + Ollama — fully local, no API key required.

Prerequisites:
    brew install ollama          # macOS
    ollama pull llama3.2         # or: mistral, phi3, gemma2, medllama2
    ollama serve                 # starts on http://localhost:11434

Run this example:
    python examples/ollama_usage.py
"""
import asyncio

from medguard import MedGuard
from medguard.config import LLMConfig, MedGuardConfig


def make_ollama_guard(model: str = "llama3.2") -> MedGuard:
    """Build a MedGuard instance backed by a local Ollama model."""
    config = MedGuardConfig(
        llm=LLMConfig(
            provider="ollama",
            model=model,
            base_url="http://localhost:11434/v1",
        )
    )
    return MedGuard(config=config)


async def main():
    mg = make_ollama_guard(model="llama3.2")

    print("=== PHI check (no LLM call) ===")
    result = mg.check(
        "Patient Jane Doe, SSN 987-65-4321, is taking warfarin and ibuprofen."
    )
    print(f"PHI detected: {result.phi_result.phi_detected}")
    print(f"Redacted: {result.phi_result.processed}")
    if result.drug_result and result.drug_result.interactions:
        print(f"Drug warning: {result.drug_result.highest_severity}")
    print()

    print("=== Guardrailed chat via Ollama ===")
    print("(make sure `ollama serve` is running)")
    try:
        response = await mg.achat(
            "What are the common side effects of metformin for type 2 diabetes?"
        )
        print("Response:", response)
    except Exception as e:
        print(f"Ollama not reachable: {e}")
        print("Start Ollama with: ollama serve")


if __name__ == "__main__":
    asyncio.run(main())
