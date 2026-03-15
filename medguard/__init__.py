"""
medguard — Healthcare-specific LLM guardrails middleware.

Quick start:

    from medguard import MedGuard

    mg = MedGuard()

    # Check text for safety issues
    result = mg.check("Patient SSN 123-45-6789 is taking warfarin and aspirin")
    print(result.blocked, result.warnings)

    # Full guardrailed chat
    import asyncio
    response = asyncio.run(mg.achat("What are side effects of metformin?"))

    # Start the FastAPI server
    app = mg.create_app()
"""
from medguard.core import MedGuard
from medguard.config import MedGuardConfig, get_config
from medguard.guardrails.pipeline import PipelineContext

__version__ = "0.1.0"
__all__ = ["MedGuard", "MedGuardConfig", "get_config", "PipelineContext"]
