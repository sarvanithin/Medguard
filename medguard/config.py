"""
medguard configuration management.

Config is loaded from ~/.medguard/config.json and can be overridden via
environment variables with the MEDGUARD_ prefix and __ as nested delimiter.

Example env overrides for Docker:
    MEDGUARD_GUARDRAILS__PHI_DETECTION__ENABLED=true
    MEDGUARD_GUARDRAILS__DRUG_SAFETY__SEVERITY_THRESHOLD=high
    MEDGUARD_LLM__PROVIDER=anthropic
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PHIConfig(BaseModel):
    enabled: bool = True
    mode: Literal["redact", "flag", "block"] = "redact"
    engine: Literal["regex", "presidio", "aws"] = "regex"
    redaction_placeholder: str = "[REDACTED]"
    entities: list[str] = Field(
        default=["SSN", "DOB", "PHONE", "EMAIL", "MRN", "ZIP", "ADDRESS", "NAME_LABELED"]
    )


class DrugSafetyConfig(BaseModel):
    enabled: bool = True
    severity_threshold: Literal["low", "moderate", "high", "contraindicated"] = "moderate"
    use_openfda: bool = True
    use_rxnorm: bool = True
    use_static_fallback: bool = True
    cache_ttl_seconds: int = 3600
    api_timeout_seconds: float = 5.0


class ScopeConfig(BaseModel):
    enabled: bool = True
    action: Literal["warn", "block"] = "warn"


class HallucinationConfig(BaseModel):
    enabled: bool = True
    confidence_threshold: float = 0.7
    check_drug_names: bool = True
    check_dosages: bool = True
    check_confident_claims: bool = True


class GuardrailsConfig(BaseModel):
    phi_detection: PHIConfig = Field(default_factory=PHIConfig)
    drug_safety: DrugSafetyConfig = Field(default_factory=DrugSafetyConfig)
    scope_enforcement: ScopeConfig = Field(default_factory=ScopeConfig)
    hallucination_detection: HallucinationConfig = Field(default_factory=HallucinationConfig)


class LLMConfig(BaseModel):
    provider: Literal["anthropic", "openai", "custom"] = "anthropic"
    model: str = "claude-haiku-4-5-20251001"
    api_key_env: str = "ANTHROPIC_API_KEY"
    base_url: str | None = None
    timeout_seconds: float = 30.0
    max_tokens: int = 2048
    system_prompt: str = (
        "You are a helpful medical information assistant. "
        "Provide accurate, evidence-based information. "
        "Always recommend consulting a healthcare professional for personal medical decisions."
    )


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    cors_origins: list[str] = ["*"]
    log_level: str = "INFO"
    log_requests: bool = True


class MedGuardConfig(BaseSettings):
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    model_config = SettingsConfigDict(
        env_prefix="MEDGUARD_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @classmethod
    def load(cls, path: Path | None = None) -> MedGuardConfig:
        """Load config from JSON file, then apply env var overrides."""
        config_path = path or _default_config_path()
        if config_path.exists():
            raw = json.loads(config_path.read_text())
            return cls.model_validate(raw)
        return cls()

    def save(self, path: Path | None = None) -> None:
        """Persist current config to JSON file."""
        config_path = path or _default_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            self.model_dump_json(indent=2, exclude_none=True)
        )


def _default_config_path() -> Path:
    env_path = __import__("os").environ.get("MEDGUARD_CONFIG")
    if env_path:
        return Path(env_path)
    return Path.home() / ".medguard" / "config.json"


def _ensure_config_dir() -> None:
    config_dir = Path.home() / ".medguard"
    config_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = config_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_config() -> MedGuardConfig:
    _ensure_config_dir()
    return MedGuardConfig.load()


def reset_config_cache() -> None:
    """Call this in tests to reset the singleton."""
    get_config.cache_clear()
