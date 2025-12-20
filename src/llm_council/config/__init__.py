"""Configuration module for LLM Council."""

from llm_council.config.models import (
    DEFAULT_COUNCIL_MODELS,
    ModelPack,
    get_council_models,
    get_model_for_pack,
)

__all__ = [
    "DEFAULT_COUNCIL_MODELS",
    "ModelPack",
    "get_council_models",
    "get_model_for_pack",
]
