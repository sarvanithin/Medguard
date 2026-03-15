"""
Protocol interfaces for medguard extension points.

Third-party plugins implement these protocols and register via pyproject.toml
entry points to add custom PHI engines, interaction sources, etc.

Example pyproject.toml entry point registration:
    [project.entry-points."medguard.phi_engines"]
    my_engine = "my_package.phi:MyPHIEngine"

    [project.entry-points."medguard.interaction_sources"]
    drugbank = "medguard_drugbank:DrugBankClient"
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from medguard.guardrails.phi import PHIMatch
    from medguard.knowledge.openfda import DrugInteraction


@runtime_checkable
class PHIEngineProtocol(Protocol):
    """Protocol for PHI detection engines."""

    def analyze(self, text: str, entities: list[str]) -> list[PHIMatch]:
        """
        Detect PHI in text.

        Args:
            text: Input text to scan.
            entities: List of entity type names to detect (e.g., ["SSN", "DOB"]).

        Returns:
            List of PHIMatch objects for each detected entity.
        """
        ...


@runtime_checkable
class ScopeClassifierProtocol(Protocol):
    """Protocol for clinical scope classifiers."""

    def classify(self, text: str) -> tuple[str, float]:
        """
        Classify the scope category of a text input.

        Returns:
            Tuple of (category_name, confidence_score).
        """
        ...


@runtime_checkable
class InteractionSourceProtocol(Protocol):
    """Protocol for drug interaction data sources."""

    async def get_drug_interactions(
        self, drug_a: str, drug_b: str
    ) -> DrugInteraction | None:
        """
        Look up interaction between two drugs.

        Args:
            drug_a: First drug name or RxCUI.
            drug_b: Second drug name or RxCUI.

        Returns:
            DrugInteraction if an interaction is found, None otherwise.
        """
        ...


@runtime_checkable
class LLMCallerProtocol(Protocol):
    """Protocol for LLM provider adapters."""

    async def call(self, prompt: str) -> str:
        """Send a prompt and return the full response text."""
        ...

    async def call_stream(self, prompt: str):
        """Send a prompt and yield response tokens as an async iterator."""
        ...
