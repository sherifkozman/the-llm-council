"""Compatibility wrapper for the renamed Gemini API provider."""

from __future__ import annotations

from llm_council.providers.gemini import (
    DEFAULT_MODEL,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_TIMEOUT_MS,
    LEGACY_MODEL_PREFIXES,
    STRUCTURED_OUTPUT_MODEL_PREFIXES,
    GeminiProvider,
    GoogleProvider,
    _strip_schema_meta_fields,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_RETRY_ATTEMPTS",
    "DEFAULT_TIMEOUT_MS",
    "LEGACY_MODEL_PREFIXES",
    "STRUCTURED_OUTPUT_MODEL_PREFIXES",
    "GeminiProvider",
    "GoogleProvider",
    "_strip_schema_meta_fields",
]
