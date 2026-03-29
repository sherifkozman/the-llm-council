"""
Gemini API provider adapter.

Direct integration with the Gemini API via Google's ``google-genai`` SDK.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any, ClassVar

from llm_council.providers.base import (
    DoctorResult,
    GenerateRequest,
    GenerateResponse,
    ProviderAdapter,
    ProviderCapabilities,
)

DEFAULT_MODEL = "gemini-3.1-pro-preview"
ENV_MODEL = "GEMINI_MODEL"
LEGACY_ENV_MODEL = "GOOGLE_MODEL"
ENV_TIMEOUT_MS = "GEMINI_TIMEOUT_MS"
LEGACY_ENV_TIMEOUT_MS = "GOOGLE_TIMEOUT_MS"
ENV_RETRY_ATTEMPTS = "GEMINI_RETRY_ATTEMPTS"
LEGACY_ENV_RETRY_ATTEMPTS = "GOOGLE_RETRY_ATTEMPTS"
DEFAULT_TIMEOUT_MS = 15000
DEFAULT_RETRY_ATTEMPTS = 1

# Model prefixes that support structured output with response_schema
# See: https://ai.google.dev/gemini-api/docs/structured-output
# Gemini 2.0+ models support structured output
STRUCTURED_OUTPUT_MODEL_PREFIXES = (
    # Gemini 3.x family (December 2025+)
    "gemini-3",  # All Gemini 3.x models (gemini-3-pro, gemini-3-flash, etc.)
    "gemini-3.0",  # Explicit 3.0 version
    "gemini-3-preview",  # Gemini 3 preview models
    # Gemini 2.x family
    "gemini-2.5",  # Gemini 2.5 (full support)
    "gemini-2.0",  # Gemini 2.0 (full support)
    "gemini-2",  # Catch-all for Gemini 2.x
    # Experimental/preview models
    "gemini-exp",  # Experimental models (gemini-exp-1206, etc.)
)

# Gemini 1.5 and older models only support simple JSON mode (no schema)
LEGACY_MODEL_PREFIXES = (
    "gemini-1.5",
    "gemini-1.0",
    "gemini-pro",  # Original Gemini Pro (1.0)
)


def _strip_schema_meta_fields(
    schema: dict[str, Any], *, _inside_properties: bool = False
) -> dict[str, Any]:
    """Strip fields from JSON schema that Google's SDK doesn't accept.

    Google's protos.Schema only accepts a limited subset of JSON Schema fields:
    - type, properties, required, items, enum, description

    It rejects standard JSON Schema fields like $schema, title, additionalProperties.
    This function recursively removes unsupported fields.

    IMPORTANT: "title" is only stripped when it's a schema meta field, NOT when it's
    a property name inside "properties". For example:
    - {"title": "MySchema", "type": "object"} -> "title" is stripped (meta field)
    - {"properties": {"title": {"type": "string"}}} -> "title" is kept (property name)

    Args:
        schema: The original JSON schema.
        _inside_properties: Internal flag - True when processing children of "properties".

    Returns:
        A new schema with only Google-supported fields.
    """
    # Fields NOT supported by Google's Schema protobuf
    # Note: "title" is handled specially - only stripped at schema level, not as property name
    unsupported_fields = {
        "$schema",
        "$id",
        "$ref",
        "$comment",
        "additionalProperties",
        "default",
        "examples",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "pattern",
        "format",
        # Array validation fields (Issue #14)
        "minItems",
        "maxItems",
        "uniqueItems",
    }

    result: dict[str, Any] = {}

    for key, value in schema.items():
        # Skip unsupported fields
        if key in unsupported_fields:
            continue

        # "title" is only a meta field at the schema level, not inside properties
        # e.g., {"properties": {"title": {"type": "string"}}} - "title" is a property name
        if key == "title" and not _inside_properties:
            continue

        if key == "properties" and isinstance(value, dict):
            # Process properties - children are property names, not schema fields
            result[key] = {
                prop_name: _strip_schema_meta_fields(prop_schema, _inside_properties=True)
                if isinstance(prop_schema, dict)
                else prop_schema
                for prop_name, prop_schema in value.items()
            }
        elif isinstance(value, dict):
            # Recursively process nested objects
            result[key] = _strip_schema_meta_fields(value, _inside_properties=False)
        elif isinstance(value, list):
            # Process arrays that might contain schemas
            result[key] = [
                _strip_schema_meta_fields(item, _inside_properties=False)
                if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            result[key] = value

    return result


logger = logging.getLogger(__name__)


class GeminiProvider(ProviderAdapter):
    """Gemini API provider adapter.

    Direct access to Gemini models via the Gemini API.
    Uses the new google-genai SDK.

    Environment variables:
        GOOGLE_API_KEY or GEMINI_API_KEY: Required. Your Google AI API key.

    Requires the 'gemini' extra:
        pip install the-llm-council[gemini]
    """

    name: ClassVar[str] = "gemini"
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        streaming=True,
        tool_use=True,
        structured_output=True,
        multimodal=True,
        max_tokens=8192,
    )

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str | None = None,
        timeout_ms: int | None = None,
        retry_attempts: int | None = None,
    ) -> None:
        """Initialize the Gemini API provider.

        Args:
            api_key: Google AI API key. Falls back to GOOGLE_API_KEY or GEMINI_API_KEY env var.
            default_model: Default model to use if not specified in request.
        """
        self._api_key = (
            api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        )
        self._default_model = (
            default_model
            or os.environ.get(ENV_MODEL)
            or os.environ.get(LEGACY_ENV_MODEL)
            or DEFAULT_MODEL
        )
        env_timeout = os.environ.get(ENV_TIMEOUT_MS) or os.environ.get(LEGACY_ENV_TIMEOUT_MS)
        self._timeout_ms = (
            timeout_ms
            if timeout_ms is not None
            else int(env_timeout)
            if env_timeout is not None
            else DEFAULT_TIMEOUT_MS
        )
        env_retry_attempts = os.environ.get(ENV_RETRY_ATTEMPTS) or os.environ.get(
            LEGACY_ENV_RETRY_ATTEMPTS
        )
        self._retry_attempts = (
            retry_attempts
            if retry_attempts is not None
            else int(env_retry_attempts)
            if env_retry_attempts is not None
            else DEFAULT_RETRY_ATTEMPTS
        )
        self._client: Any = None

    def _client_timeout_ms(self, request: GenerateRequest | None = None) -> int:
        """Resolve the SDK timeout for a request."""
        if request and request.timeout_seconds is not None:
            return max(int(request.timeout_seconds * 1000), 1)
        return self._timeout_ms

    def _build_client(self, *, timeout_ms: int) -> Any:
        """Build a Google GenAI client with the requested timeout."""
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "The 'google-genai' package is required for the Gemini provider. "
                "Install it with: pip install the-llm-council[gemini]"
            ) from e

        if not self._api_key:
            raise ValueError(
                "Google AI API key not configured. "
                "Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable or pass api_key."
            )

        return genai.Client(
            api_key=self._api_key,
            http_options=genai.types.HttpOptions(
                timeout=timeout_ms,
                retry_options=genai.types.HttpRetryOptions(
                    attempts=self._retry_attempts,
                ),
            ),
        )

    def _get_client(self, *, timeout_ms: int | None = None) -> Any:
        """Get or create the Google GenAI client."""
        resolved_timeout = timeout_ms or self._timeout_ms
        if resolved_timeout != self._timeout_ms:
            return self._build_client(timeout_ms=resolved_timeout)
        if self._client is None:
            self._client = self._build_client(timeout_ms=resolved_timeout)
        return self._client

    async def generate(
        self, request: GenerateRequest
    ) -> GenerateResponse | AsyncIterator[GenerateResponse]:
        """Generate a response using the Gemini API."""
        client = self._get_client(timeout_ms=self._client_timeout_ms(request))
        model = request.model or self._default_model

        # Build content - new SDK uses simple string or list format
        contents: str | list[dict[str, Any]]
        if request.messages:
            # Convert messages to the format expected by the new SDK
            contents = []
            for m in request.messages:
                role = "user" if m.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": m.content}]})
        elif request.prompt:
            contents = request.prompt
        else:
            raise ValueError("Either 'messages' or 'prompt' must be provided")

        # Build config dict
        config: dict[str, Any] = {}
        if request.max_tokens is not None:
            config["max_output_tokens"] = request.max_tokens
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.top_p is not None:
            config["top_p"] = request.top_p
        if request.stop:
            config["stop_sequences"] = list(request.stop)

        # Handle structured output - requires response_mime_type and response_schema
        # See: https://ai.google.dev/gemini-api/docs/structured-output
        if request.structured_output:
            if self._model_supports_structured_output(model):
                config["response_mime_type"] = "application/json"
                # Strip $schema and other meta fields Google doesn't accept
                config["response_schema"] = _strip_schema_meta_fields(
                    dict(request.structured_output.json_schema)
                )
            elif self._is_legacy_model(model):
                # Fall back to simple JSON mode for older models (no schema enforcement)
                config["response_mime_type"] = "application/json"
            # else: model doesn't support structured output, skip

        # Handle reasoning/thinking configuration
        # Gemini 3.x uses thinking_level, Gemini 2.5 uses thinking_budget
        if request.reasoning and request.reasoning.enabled:
            if request.reasoning.thinking_level:
                # Gemini 3.x style: minimal, low, medium, high
                config["thinking_config"] = {
                    "thinking_level": request.reasoning.thinking_level.upper(),
                }
            elif request.reasoning.budget_tokens:
                # Gemini 2.5 style: token budget (max 24576)
                max_budget = 24576
                budget = min(request.reasoning.budget_tokens, max_budget)
                if request.reasoning.budget_tokens > max_budget:
                    logger.warning(
                        "Google thinking_budget capped from %d to %d (provider maximum)",
                        request.reasoning.budget_tokens,
                        max_budget,
                    )
                config["thinking_config"] = {
                    "thinking_budget": budget,
                }
            else:
                # Default: medium thinking level
                config["thinking_config"] = {
                    "thinking_level": "MEDIUM",
                }

        if request.stream:
            return self._generate_stream(client, model, contents, config)

        # Use async client for generate_content
        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config if config else None,
        )
        return self._parse_response(response)

    async def _generate_stream(
        self, client: Any, model: str, contents: Any, config: dict[str, Any]
    ) -> AsyncIterator[GenerateResponse]:
        """Stream responses from the Gemini API."""
        # The new SDK's stream method is a coroutine that returns an async iterator
        stream = await client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config if config else None,
        )
        async for chunk in stream:
            if chunk.text:
                yield GenerateResponse(text=chunk.text, content=chunk.text)

    def _parse_response(self, response: Any) -> GenerateResponse:
        """Parse Gemini API response."""
        text = ""
        try:
            text = response.text
        except Exception:
            # Try to extract text from parts if .text fails
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content:
                    parts = candidate.content.parts
                    if parts:
                        text = "".join(part.text for part in parts if hasattr(part, "text"))

        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(um, "candidates_token_count", 0) or 0,
                "total_tokens": getattr(um, "total_token_count", 0) or 0,
            }

        finish_reason = None
        if hasattr(response, "candidates") and response.candidates:
            fr = getattr(response.candidates[0], "finish_reason", None)
            if fr:
                finish_reason = str(fr)

        return GenerateResponse(
            text=text,
            content=text,
            usage=usage,
            finish_reason=finish_reason,
            raw=response,
        )

    def _model_supports_structured_output(self, model: str) -> bool:
        """Check if a specific model supports structured output with response_schema.

        Gemini 2.0+ models support structured output with JSON schema.
        Gemini 1.5 and older only support simple JSON mode.

        Args:
            model: The model identifier to check.

        Returns:
            True if the model supports response_schema in generation_config.
        """
        # Check if model starts with any supported prefix (Gemini 2.0+)
        return any(model.startswith(prefix) for prefix in STRUCTURED_OUTPUT_MODEL_PREFIXES)

    def _is_legacy_model(self, model: str) -> bool:
        """Check if a model is a legacy model that only supports JSON mode.

        Args:
            model: The model identifier to check.

        Returns:
            True if the model only supports simple JSON mode (no schema).
        """
        return any(model.startswith(prefix) for prefix in LEGACY_MODEL_PREFIXES)

    async def supports(self, capability: str) -> bool:
        """Check if the provider supports a capability."""
        if not self.supports_capability_name(capability):
            return False
        return getattr(self.capabilities, capability, False)

    async def doctor(self) -> DoctorResult:
        """Perform a health check on the Gemini API."""
        start_time = time.time()

        if not self._api_key:
            return DoctorResult(
                ok=False,
                message="GOOGLE_API_KEY environment variable not set",
                details={"error": "missing_api_key"},
            )

        try:
            from google import genai
        except ImportError:
            return DoctorResult(
                ok=False,
                message="google-genai not installed. Run: pip install the-llm-council[gemini]",
                details={"error": "missing_package"},
            )

        try:
            client = self._get_client()
            # List models to verify API key works
            # The new SDK uses client.models.list()
            list(client.models.list())
            latency_ms = (time.time() - start_time) * 1000

            return DoctorResult(
                ok=True,
                message="Gemini API is accessible",
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return DoctorResult(
                ok=False,
                message=f"API error: {str(e)}",
                latency_ms=latency_ms,
                details={"error": str(e)},
            )


def _register() -> None:
    """Register the Gemini API provider with the global registry."""
    from llm_council.providers.registry import get_registry

    registry = get_registry()
    with contextlib.suppress(ValueError):
        registry.register_provider("gemini", GeminiProvider)


_register()

# Backwards-compatible import alias. Provider name compatibility is handled in
# the registry aliases, but older import sites may still import GoogleProvider.
GoogleProvider = GeminiProvider
