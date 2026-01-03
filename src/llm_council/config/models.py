"""
Model configuration for multi-model council.

This module handles the configuration of multiple LLM models for council runs.
Models can be configured via environment variables or passed directly.

Environment Variables:
    COUNCIL_MODELS: Comma-separated list of OpenRouter model IDs
        Example: "anthropic/claude-3.5-sonnet,openai/gpt-4o,google/gemini-pro"

    Model pack overrides (optional):
        COUNCIL_MODEL_FAST: Fast model for quick tasks (default: claude-3-haiku)
        COUNCIL_MODEL_REASONING: Deep reasoning model (default: claude-opus-4-5)
        COUNCIL_MODEL_CODE: Code specialist model (default: gpt-5.1)
        COUNCIL_MODEL_CRITIC: Adversarial critic model (default: claude-sonnet-4-5)
        COUNCIL_MODEL_GROUNDED: Grounded/RAG model (default: gemini-3-pro-preview)
"""

from __future__ import annotations

import os
from enum import Enum
from typing import ClassVar


class ModelPack(str, Enum):
    """Model pack types for different task categories.

    These map to the model_pack values used in subagent YAML configurations.
    """

    # Core packs
    FAST = "fast"  # Quick classification, routing (YAML: fast_generator)
    REASONING = "reasoning"  # Deep analysis, planning (YAML: deep_reasoner)
    CODE = "code"  # Code generation, implementation (YAML: code_specialist_normal)
    CODE_COMPLEX = "code_complex"  # Complex refactoring (YAML: code_specialist_complex)
    CRITIC = "critic"  # Adversarial review, critique (YAML: harsh_critic)
    GROUNDED = "grounded"  # RAG, web search, research (YAML: grounded)
    DEFAULT = "default"  # General purpose fallback


# Mapping from YAML model_pack strings to ModelPack enum
# This bridges the gap between subagent YAML configs and Python code
YAML_TO_MODEL_PACK: dict[str, ModelPack] = {
    # YAML string -> ModelPack enum
    "fast_generator": ModelPack.FAST,
    "deep_reasoner": ModelPack.REASONING,
    "code_specialist_normal": ModelPack.CODE,
    "code_specialist_complex": ModelPack.CODE_COMPLEX,
    "harsh_critic": ModelPack.CRITIC,
    "grounded": ModelPack.GROUNDED,
    # Direct enum value names also work
    "fast": ModelPack.FAST,
    "reasoning": ModelPack.REASONING,
    "code": ModelPack.CODE,
    "code_complex": ModelPack.CODE_COMPLEX,
    "critic": ModelPack.CRITIC,
    "default": ModelPack.DEFAULT,
}


def resolve_model_pack(yaml_value: str) -> ModelPack:
    """Resolve a YAML model_pack string to a ModelPack enum.

    Args:
        yaml_value: The model_pack value from a subagent YAML file.

    Returns:
        The corresponding ModelPack enum value.

    Raises:
        ValueError: If the yaml_value is not a recognized model pack.
    """
    pack = YAML_TO_MODEL_PACK.get(yaml_value.lower())
    if pack is None:
        valid = ", ".join(sorted(YAML_TO_MODEL_PACK.keys()))
        raise ValueError(f"Unknown model_pack '{yaml_value}'. Valid values: {valid}")
    return pack


# Default models for each pack (OpenRouter format) - January 2026
DEFAULT_MODEL_PACKS: dict[ModelPack, str] = {
    ModelPack.FAST: "anthropic/claude-3-5-haiku",
    ModelPack.REASONING: "anthropic/claude-opus-4-5",
    ModelPack.CODE: "openai/gpt-5.1",
    ModelPack.CODE_COMPLEX: "anthropic/claude-opus-4-5",  # Use Opus for complex refactors
    ModelPack.CRITIC: "anthropic/claude-sonnet-4-5",
    ModelPack.GROUNDED: "google/gemini-3-pro-preview",  # Gemini for grounded/search tasks
    ModelPack.DEFAULT: "anthropic/claude-opus-4-5",
}

# Default council models for multi-model runs - January 2026
DEFAULT_COUNCIL_MODELS: list[str] = [
    "anthropic/claude-opus-4-5",
    "openai/gpt-5.1",
    "google/gemini-3-pro-preview",
]

# Environment variable names
ENV_COUNCIL_MODELS = "COUNCIL_MODELS"
ENV_MODEL_FAST = "COUNCIL_MODEL_FAST"
ENV_MODEL_REASONING = "COUNCIL_MODEL_REASONING"
ENV_MODEL_CODE = "COUNCIL_MODEL_CODE"
ENV_MODEL_CODE_COMPLEX = "COUNCIL_MODEL_CODE_COMPLEX"
ENV_MODEL_CRITIC = "COUNCIL_MODEL_CRITIC"
ENV_MODEL_GROUNDED = "COUNCIL_MODEL_GROUNDED"


class ModelConfig:
    """Configuration for council models."""

    _instance: ClassVar[ModelConfig | None] = None

    def __init__(self) -> None:
        self._models: list[str] | None = None
        self._pack_overrides: dict[ModelPack, str] = {}
        self._load_from_env()

    @classmethod
    def get_instance(cls) -> ModelConfig:
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Load council models
        models_env = os.environ.get(ENV_COUNCIL_MODELS)
        if models_env:
            self._models = [m.strip() for m in models_env.split(",") if m.strip()]

        # Load model pack overrides
        pack_env_map = {
            ModelPack.FAST: ENV_MODEL_FAST,
            ModelPack.REASONING: ENV_MODEL_REASONING,
            ModelPack.CODE: ENV_MODEL_CODE,
            ModelPack.CODE_COMPLEX: ENV_MODEL_CODE_COMPLEX,
            ModelPack.CRITIC: ENV_MODEL_CRITIC,
            ModelPack.GROUNDED: ENV_MODEL_GROUNDED,
        }
        for pack, env_var in pack_env_map.items():
            value = os.environ.get(env_var)
            if value:
                self._pack_overrides[pack] = value.strip()

    def get_council_models(self) -> list[str]:
        """Get the list of models for council runs.

        Returns:
            List of OpenRouter model IDs. If COUNCIL_MODELS env var is not set,
            returns a single-element list with the default model.
        """
        if self._models:
            return list(self._models)
        # Return single default model if not configured
        return [DEFAULT_MODEL_PACKS[ModelPack.DEFAULT]]

    def get_model_for_pack(self, pack: ModelPack) -> str:
        """Get the model for a specific pack.

        Args:
            pack: The model pack type.

        Returns:
            OpenRouter model ID for the pack.
        """
        # Check for override first
        if pack in self._pack_overrides:
            return self._pack_overrides[pack]
        # Fall back to default
        return DEFAULT_MODEL_PACKS.get(pack, DEFAULT_MODEL_PACKS[ModelPack.DEFAULT])

    def is_multi_model_enabled(self) -> bool:
        """Check if multi-model council is enabled.

        Returns:
            True if COUNCIL_MODELS is set with multiple models.
        """
        return self._models is not None and len(self._models) > 1


def get_council_models() -> list[str]:
    """Get the list of models for council runs.

    Convenience function that uses the singleton ModelConfig.

    Returns:
        List of OpenRouter model IDs.
    """
    return ModelConfig.get_instance().get_council_models()


def get_model_for_pack(pack: ModelPack) -> str:
    """Get the model for a specific pack.

    Convenience function that uses the singleton ModelConfig.

    Args:
        pack: The model pack type.

    Returns:
        OpenRouter model ID for the pack.
    """
    return ModelConfig.get_instance().get_model_for_pack(pack)


def is_multi_model_enabled() -> bool:
    """Check if multi-model council is enabled.

    Convenience function that uses the singleton ModelConfig.

    Returns:
        True if COUNCIL_MODELS is set with multiple models.
    """
    return ModelConfig.get_instance().is_multi_model_enabled()


def parse_models_string(models_str: str) -> list[str]:
    """Parse a comma-separated string of model names.

    Args:
        models_str: Comma-separated model names.

    Returns:
        List of model names, trimmed and filtered.
    """
    return [m.strip() for m in models_str.split(",") if m.strip()]
