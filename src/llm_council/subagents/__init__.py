"""
Subagent configurations.

YAML-based configurations for each subagent role:
- router, planner, assessor, researcher, architect
- implementer, reviewer, test-designer, shipper, red-team

Supports per-subagent configuration for:
- Provider preferences (preferred, fallback, exclude)
- Model selection per provider
- Reasoning/thinking budgets
"""

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

# ============================================================================
# Subagent Configuration Models (Issue #11)
# ============================================================================


class ProviderPreferences(BaseModel):
    """Provider preference configuration for a subagent."""

    preferred: list[str] = Field(
        default_factory=list,
        description="Ordered list of preferred providers.",
    )
    fallback: list[str] = Field(
        default_factory=list,
        description="Providers to use if preferred are unavailable.",
    )
    exclude: list[str] = Field(
        default_factory=list,
        description="Providers to never use for this subagent.",
    )


class ModelOverrides(BaseModel):
    """Per-provider model overrides for a subagent."""

    openai: str | None = Field(
        default=None,
        description="OpenAI model to use (e.g., 'gpt-5.1', 'o3-mini').",
    )
    anthropic: str | None = Field(
        default=None,
        description="Anthropic model to use (e.g., 'claude-opus-4-5').",
    )
    google: str | None = Field(
        default=None,
        description="Google model to use (e.g., 'gemini-3-flash-preview').",
    )
    openrouter: str | None = Field(
        default=None,
        description="OpenRouter model ID (e.g., 'openai/gpt-5.1').",
    )

    def get_for_provider(self, provider_name: str) -> str | None:
        """Get model override for a specific provider."""
        return getattr(self, provider_name, None)


class ReasoningBudget(BaseModel):
    """Reasoning/thinking budget configuration for a subagent."""

    enabled: bool = Field(
        default=False,
        description="Enable reasoning/thinking mode.",
    )
    effort: Literal["low", "medium", "high", "none"] | None = Field(
        default=None,
        description="OpenAI o-series reasoning effort level.",
    )
    budget_tokens: int | None = Field(
        default=None,
        ge=1024,
        le=128000,
        description="Anthropic thinking budget in tokens (1024-128000).",
    )
    thinking_level: Literal["minimal", "low", "medium", "high"] | None = Field(
        default=None,
        description="Google Gemini 3.x thinking level.",
    )


SUBAGENTS_DIR = Path(__file__).parent

# Strict allowlist pattern: lowercase alphanumeric, hyphens, underscores only
_VALID_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _validate_name(name: str, resource_type: str = "subagent") -> None:
    """Validate resource name against strict allowlist to prevent path traversal."""
    if not name:
        raise ValueError(f"{resource_type} name cannot be empty")
    if not _VALID_NAME_PATTERN.match(name):
        raise ValueError(
            f"Invalid {resource_type} name '{name}': must match pattern "
            f"'^[a-z0-9][a-z0-9_-]*$' (lowercase alphanumeric, hyphens, underscores)"
        )


def _ensure_path_containment(path: Path, base_dir: Path, resource_type: str) -> None:
    """Ensure resolved path stays within the base directory."""
    resolved = path.resolve()
    base_resolved = base_dir.resolve()
    try:
        resolved.relative_to(base_resolved)
    except ValueError:
        raise ValueError(f"{resource_type} path escapes allowed directory: {path}")


def load_subagent(name: str) -> dict[str, Any]:
    """Load a subagent configuration by name.

    Args:
        name: Subagent name (must match ^[a-z0-9][a-z0-9_-]*$)

    Returns:
        Parsed YAML configuration dict

    Raises:
        ValueError: If name is invalid or path escapes subagents directory
        FileNotFoundError: If config file doesn't exist
    """
    _validate_name(name, "subagent")
    config_path = SUBAGENTS_DIR / f"{name}.yaml"
    _ensure_path_containment(config_path, SUBAGENTS_DIR, "Subagent config")

    if not config_path.exists():
        raise FileNotFoundError(f"Subagent config not found: {name}")
    with open(config_path) as f:
        loaded = yaml.safe_load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"Subagent config for '{name}' must be a YAML dictionary")
        return loaded


def list_subagents() -> list[str]:
    """List available subagent names."""
    return [p.stem for p in SUBAGENTS_DIR.glob("*.yaml")]


def get_provider_preferences(config: dict[str, Any]) -> ProviderPreferences | None:
    """Extract provider preferences from a subagent config.

    Args:
        config: Loaded subagent YAML configuration.

    Returns:
        ProviderPreferences if 'providers' key exists, None otherwise.
    """
    providers_data = config.get("providers")
    if providers_data:
        return ProviderPreferences(**providers_data)
    return None


def get_model_overrides(config: dict[str, Any]) -> ModelOverrides | None:
    """Extract model overrides from a subagent config.

    Args:
        config: Loaded subagent YAML configuration.

    Returns:
        ModelOverrides if 'models' key exists, None otherwise.
    """
    models_data = config.get("models")
    if models_data:
        return ModelOverrides(**models_data)
    return None


def get_reasoning_budget(config: dict[str, Any]) -> ReasoningBudget | None:
    """Extract reasoning budget from a subagent config.

    Args:
        config: Loaded subagent YAML configuration.

    Returns:
        ReasoningBudget if 'reasoning' key exists, None otherwise.
    """
    reasoning_data = config.get("reasoning")
    if reasoning_data:
        return ReasoningBudget(**reasoning_data)
    return None


__all__ = [
    "load_subagent",
    "list_subagents",
    "SUBAGENTS_DIR",
    # Issue #11: Per-subagent configuration
    "ProviderPreferences",
    "ModelOverrides",
    "ReasoningBudget",
    "get_provider_preferences",
    "get_model_overrides",
    "get_reasoning_budget",
]
