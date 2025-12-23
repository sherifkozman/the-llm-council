"""
Model configuration for multi-model council.

This module handles the configuration of multiple LLM models for council runs.
Models can be configured via environment variables or passed directly.

Environment Variables:
    COUNCIL_MODELS: Comma-separated list of OpenRouter model IDs
        Example: "anthropic/claude-3.5-sonnet,openai/gpt-4o,google/gemini-pro"

    Model pack overrides (optional):
        COUNCIL_MODEL_FAST: Fast model for quick tasks (default: claude-3-haiku)
        COUNCIL_MODEL_REASONING: Deep reasoning model (default: claude-3-opus)
        COUNCIL_MODEL_CODE: Code specialist model (default: gpt-4o)
        COUNCIL_MODEL_CRITIC: Adversarial critic model (default: claude-3.5-sonnet)
"""

from __future__ import annotations

import os
from enum import Enum
from typing import ClassVar


class ModelPack(str, Enum):
    """Model pack types for different task categories."""

    FAST = "fast"  # Quick classification, routing
    REASONING = "reasoning"  # Deep analysis, planning
    CODE = "code"  # Code generation, implementation
    CRITIC = "critic"  # Adversarial review, critique
    DEFAULT = "default"  # General purpose


# Default models for each pack (OpenRouter format) - December 2025
DEFAULT_MODEL_PACKS: dict[ModelPack, str] = {
    ModelPack.FAST: "anthropic/claude-3-5-haiku",
    ModelPack.REASONING: "anthropic/claude-opus-4-5",
    ModelPack.CODE: "openai/gpt-5.1",
    ModelPack.CRITIC: "anthropic/claude-sonnet-4-5",
    ModelPack.DEFAULT: "anthropic/claude-opus-4-5",
}

# Default council models for multi-model runs - December 2025
DEFAULT_COUNCIL_MODELS: list[str] = [
    "anthropic/claude-opus-4-5",
    "openai/gpt-5.1",
    "google/gemini-3-flash-preview",
]

# Environment variable names
ENV_COUNCIL_MODELS = "COUNCIL_MODELS"
ENV_MODEL_FAST = "COUNCIL_MODEL_FAST"
ENV_MODEL_REASONING = "COUNCIL_MODEL_REASONING"
ENV_MODEL_CODE = "COUNCIL_MODEL_CODE"
ENV_MODEL_CRITIC = "COUNCIL_MODEL_CRITIC"


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
            ModelPack.CRITIC: ENV_MODEL_CRITIC,
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
