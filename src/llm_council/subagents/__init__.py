"""
Subagent configurations.

YAML-based configurations for each subagent role (v0.5.0):
- Core: drafter, critic, synthesizer, researcher, planner, router
- Legacy (deprecated): implementer, architect, reviewer, etc.

Supports per-subagent configuration for:
- Provider preferences (preferred, fallback, exclude)
- Model selection per provider
- Reasoning/thinking budgets
- Model packs (maps to specific models)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, Field

from llm_council.config.models import (
    ModelPack,
    get_model_for_pack,
    normalize_model_pack,
    resolve_model_pack,
)

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
        description="OpenAI model to use (e.g., 'gpt-5.4', 'o3-mini').",
    )
    anthropic: str | None = Field(
        default=None,
        description="Anthropic model to use (e.g., 'claude-opus-4-6').",
    )
    google: str | None = Field(
        default=None,
        description="Google model to use (e.g., 'gemini-3-flash-preview').",
    )
    openrouter: str | None = Field(
        default=None,
        description="OpenRouter model ID (e.g., 'openai/gpt-5.4').",
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


def _merge_config_dicts(
    base: dict[str, Any] | None, override: dict[str, Any] | None
) -> dict[str, Any]:
    """Merge shallow config dictionaries with override precedence."""

    merged: dict[str, Any] = {}
    if isinstance(base, dict):
        merged.update(base)
    if isinstance(override, dict):
        merged.update(override)
    return merged


def resolve_mode(config: dict[str, Any], mode: str | None = None) -> str | None:
    """Resolve the active mode for a subagent configuration.

    Args:
        config: Loaded subagent YAML configuration.
        mode: Optional requested mode.

    Returns:
        Resolved mode name or None if the subagent is mode-less.

    Raises:
        ValueError: If the requested mode is not defined for the subagent.
    """

    modes = config.get("modes", {})
    if not modes:
        return mode

    if mode:
        if mode not in modes:
            valid = ", ".join(sorted(modes.keys()))
            raise ValueError(f"Invalid mode '{mode}'. Valid modes: {valid}")
        return mode

    default_mode = config.get("default_mode")
    if isinstance(default_mode, str):
        if default_mode not in modes:
            valid = ", ".join(sorted(modes.keys()))
            raise ValueError(f"Invalid default_mode '{default_mode}'. Valid modes: {valid}")
        return default_mode

    return None


def get_effective_schema(config: dict[str, Any], mode: str | None = None) -> str | None:
    """Get the effective schema name for a subagent and mode."""

    resolved_mode = resolve_mode(config, mode)
    if resolved_mode and "modes" in config:
        modes = config.get("modes")
        if isinstance(modes, dict):
            mode_config = modes.get(resolved_mode, {})
            if isinstance(mode_config, dict):
                schema = mode_config.get("schema")
                if isinstance(schema, str):
                    return schema
    schema = config.get("schema")
    return schema if isinstance(schema, str) else None


def get_effective_system_prompt(config: dict[str, Any], mode: str | None = None) -> str:
    """Get the effective system prompt, including mode-specific additions."""

    prompts = config.get("prompts", {})
    parts: list[str] = []
    system_prompt = prompts.get("system")
    if isinstance(system_prompt, str) and system_prompt.strip():
        parts.append(system_prompt.strip())

    resolved_mode = resolve_mode(config, mode)
    if resolved_mode:
        mode_prompt = prompts.get("mode_prompts", {}).get(resolved_mode)
        if isinstance(mode_prompt, str) and mode_prompt.strip():
            parts.append(mode_prompt.strip())

    return "\n\n".join(parts)


def get_provider_preferences(
    config: dict[str, Any], mode: str | None = None
) -> ProviderPreferences | None:
    """Extract provider preferences from a subagent config, including mode overrides."""

    resolved_mode = resolve_mode(config, mode)
    mode_config = get_mode_config(config, resolved_mode) if resolved_mode else {}
    providers_data = _merge_config_dicts(config.get("providers"), mode_config.get("providers"))
    if providers_data:
        return ProviderPreferences(**providers_data)
    return None


def get_model_overrides(config: dict[str, Any], mode: str | None = None) -> ModelOverrides | None:
    """Extract model overrides from a subagent config.

    Args:
        config: Loaded subagent YAML configuration.

    Returns:
        ModelOverrides if 'models' key exists, None otherwise.
    """
    resolved_mode = resolve_mode(config, mode)
    mode_config = get_mode_config(config, resolved_mode) if resolved_mode else {}
    models_data = _merge_config_dicts(config.get("models"), mode_config.get("models"))
    if models_data:
        return ModelOverrides(**models_data)
    return None


def get_reasoning_budget(config: dict[str, Any], mode: str | None = None) -> ReasoningBudget | None:
    """Extract reasoning budget from a subagent config.

    Args:
        config: Loaded subagent YAML configuration.

    Returns:
        ReasoningBudget if 'reasoning' key exists, None otherwise.
    """
    resolved_mode = resolve_mode(config, mode)
    mode_config = get_mode_config(config, resolved_mode) if resolved_mode else {}
    reasoning_data = _merge_config_dicts(config.get("reasoning"), mode_config.get("reasoning"))
    if reasoning_data:
        return ReasoningBudget(**reasoning_data)
    return None


def get_model_pack(config: dict[str, Any], mode: str | None = None) -> ModelPack:
    """Get the resolved ModelPack from a subagent config.

    Handles mode-specific model packs for consolidated agents (drafter, critic, planner).

    Args:
        config: Loaded subagent YAML configuration.
        mode: Optional mode (e.g., 'impl', 'arch', 'review', 'security').

    Returns:
        Resolved ModelPack enum value.

    Example:
        >>> config = load_subagent("drafter")
        >>> pack = get_model_pack(config, mode="arch")
        >>> pack == ModelPack.REASONING  # deep_reasoner maps to REASONING
        True
    """
    # Check for mode-specific model_pack first
    if mode and "modes" in config:
        mode_config = config["modes"].get(mode, {})
        if "model_pack" in mode_config:
            return resolve_model_pack(mode_config["model_pack"])

    # Fall back to top-level model_pack
    if "model_pack" in config:
        return resolve_model_pack(config["model_pack"])

    # Default to DEFAULT pack
    return ModelPack.DEFAULT


def get_model_for_subagent(
    config: dict[str, Any],
    mode: str | None = None,
    model_pack: str | ModelPack | None = None,
) -> str:
    """Get the default model ID for a subagent based on its model pack.

    Args:
        config: Loaded subagent YAML configuration.
        mode: Optional mode for consolidated agents.
        model_pack: Optional runtime model-pack override.

    Returns:
        OpenRouter-format model ID (e.g., 'anthropic/claude-opus-4-6').
    """
    pack = (
        normalize_model_pack(model_pack) if model_pack is not None else get_model_pack(config, mode)
    )
    return get_model_for_pack(pack)


def get_mode_config(config: dict[str, Any], mode: str) -> dict[str, Any]:
    """Get mode-specific configuration for a consolidated agent.

    Args:
        config: Loaded subagent YAML configuration.
        mode: Mode name (e.g., 'impl', 'arch', 'review').

    Returns:
        Mode configuration dict, or empty dict if mode not found.
    """
    modes = config.get("modes")
    if not isinstance(modes, dict):
        return {}
    mode_config = modes.get(mode, {})
    return cast(dict[str, Any], mode_config) if isinstance(mode_config, dict) else {}


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
    "resolve_mode",
    "get_effective_schema",
    "get_effective_system_prompt",
    # Model pack resolution (v0.5.1)
    "get_model_pack",
    "get_model_for_subagent",
    "get_mode_config",
]
