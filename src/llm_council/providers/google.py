"""
Google AI provider adapter.

Direct integration with the Google Generative AI API for Gemini models.
"""

from __future__ import annotations

import contextlib
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

DEFAULT_MODEL = "gemini-3-flash-preview"

# Model prefixes that support structured output with response_schema
# See: https://ai.google.dev/gemini-api/docs/structured-output
# Gemini 2.0+ models support structured output
STRUCTURED_OUTPUT_MODEL_PREFIXES = (
    # Gemini 3.x family (December 2025+)
    "gemini-3",           # All Gemini 3.x models (gemini-3-pro, gemini-3-flash, etc.)
    "gemini-3.0",         # Explicit 3.0 version
    "gemini-3-preview",   # Gemini 3 preview models
    # Gemini 2.x family
    "gemini-2.5",         # Gemini 2.5 (full support)
    "gemini-2.0",         # Gemini 2.0 (full support)
    "gemini-2",           # Catch-all for Gemini 2.x
    # Experimental/preview models
    "gemini-exp",         # Experimental models (gemini-exp-1206, etc.)
)

# Gemini 1.5 and older models only support simple JSON mode (no schema)
LEGACY_MODEL_PREFIXES = (
    "gemini-1.5",
    "gemini-1.0",
    "gemini-pro",  # Original Gemini Pro (1.0)
)


def _strip_schema_meta_fields(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip fields from JSON schema that Google's SDK doesn't accept.

    Google's protos.Schema only accepts a limited subset of JSON Schema fields:
    - type, properties, required, items, enum, description

    It rejects standard JSON Schema fields like $schema, title, additionalProperties.
    This function recursively removes unsupported fields.

    Args:
        schema: The original JSON schema.

    Returns:
        A new schema with only Google-supported fields.
    """
    # Fields NOT supported by Google's Schema protobuf
    unsupported_fields = {
        "$schema",
        "$id",
        "$ref",
        "$comment",
        "title",
        "additionalProperties",
        "default",
        "examples",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "pattern",
        "format",
    }

    result: dict[str, Any] = {}

    for key, value in schema.items():
        # Skip unsupported fields
        if key in unsupported_fields:
            continue

        if isinstance(value, dict):
            # Recursively process nested objects
            result[key] = _strip_schema_meta_fields(value)
        elif isinstance(value, list):
            # Process arrays that might contain schemas
            result[key] = [
                _strip_schema_meta_fields(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value

    return result


class GoogleProvider(ProviderAdapter):
    """Google AI API provider adapter.

    Direct access to Gemini models via the Google Generative AI API.

    Environment variables:
        GOOGLE_API_KEY: Required. Your Google AI API key.

    Requires the 'google' extra:
        pip install the-llm-council[google]
    """

    name: ClassVar[str] = "google"
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
    ) -> None:
        """Initialize the Google AI provider.

        Args:
            api_key: Google AI API key. Falls back to GOOGLE_API_KEY env var.
            default_model: Default model to use if not specified in request.
        """
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._default_model = default_model or DEFAULT_MODEL
        self._client: Any = None

    def _get_client(self, model: str) -> Any:
        """Get or create a Generative Model client."""
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "The 'google-generativeai' package is required for the Google provider. "
                "Install it with: pip install the-llm-council[google]"
            ) from e

        if not self._api_key:
            raise ValueError(
                "Google AI API key not configured. "
                "Set GOOGLE_API_KEY environment variable or pass api_key."
            )

        genai.configure(api_key=self._api_key)
        return genai.GenerativeModel(model)

    async def generate(
        self, request: GenerateRequest
    ) -> GenerateResponse | AsyncIterator[GenerateResponse]:
        """Generate a response using Google AI API."""
        model = request.model or self._default_model
        client = self._get_client(model)

        # Build content
        contents: list[dict[str, Any]] = []
        if request.messages:
            for m in request.messages:
                role = "user" if m.role == "user" else "model"
                contents.append({"role": role, "parts": [m.content]})
        elif request.prompt:
            contents = [{"role": "user", "parts": [request.prompt]}]
        else:
            raise ValueError("Either 'messages' or 'prompt' must be provided")

        generation_config: dict[str, Any] = {}
        if request.max_tokens is not None:
            generation_config["max_output_tokens"] = request.max_tokens
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.top_p is not None:
            generation_config["top_p"] = request.top_p
        if request.stop:
            generation_config["stop_sequences"] = list(request.stop)

        # Handle structured output - requires response_mime_type and response_schema
        # See: https://ai.google.dev/gemini-api/docs/structured-output
        if request.structured_output:
            if self._model_supports_structured_output(model):
                generation_config["response_mime_type"] = "application/json"
                # Strip $schema and other meta fields Google doesn't accept
                generation_config["response_schema"] = _strip_schema_meta_fields(
                    dict(request.structured_output.json_schema)
                )
            elif self._is_legacy_model(model):
                # Fall back to simple JSON mode for older models (no schema enforcement)
                generation_config["response_mime_type"] = "application/json"
            # else: model doesn't support structured output, skip

        # Handle reasoning/thinking configuration
        # Gemini 3.x uses thinking_level, Gemini 2.5 uses thinking_budget
        if request.reasoning and request.reasoning.enabled:
            if request.reasoning.thinking_level:
                # Gemini 3.x style: minimal, low, medium, high
                generation_config["thinking_config"] = {
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
                generation_config["thinking_config"] = {
                    "thinking_budget": budget,
                }
            else:
                # Default: medium thinking level
                generation_config["thinking_config"] = {
                    "thinking_level": "MEDIUM",
                }

        if request.stream:
            return self._generate_stream(client, contents, generation_config)

        response = await client.generate_content_async(
            contents,
            generation_config=generation_config if generation_config else None,
        )
        return self._parse_response(response)

    async def _generate_stream(
        self, client: Any, contents: Any, generation_config: dict[str, Any]
    ) -> AsyncIterator[GenerateResponse]:
        """Stream responses from Google AI API."""
        response = await client.generate_content_async(
            contents,
            generation_config=generation_config if generation_config else None,
            stream=True,
        )
        async for chunk in response:
            if chunk.text:
                yield GenerateResponse(text=chunk.text, content=chunk.text)

    def _parse_response(self, response: Any) -> GenerateResponse:
        """Parse Google AI API response."""
        text = ""
        try:
            text = response.text
        except Exception:
            if response.parts:
                text = "".join(part.text for part in response.parts if hasattr(part, "text"))

        usage = None
        if hasattr(response, "usage_metadata"):
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }

        finish_reason = None
        if response.candidates:
            finish_reason = str(response.candidates[0].finish_reason)

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
        """Perform a health check on the Google AI API."""
        start_time = time.time()

        if not self._api_key:
            return DoctorResult(
                ok=False,
                message="GOOGLE_API_KEY environment variable not set",
                details={"error": "missing_api_key"},
            )

        try:
            import google.generativeai as genai
        except ImportError:
            return DoctorResult(
                ok=False,
                message="google-generativeai package not installed. Run: pip install the-llm-council[google]",
                details={"error": "missing_package"},
            )

        try:
            genai.configure(api_key=self._api_key)
            # List models to verify API key
            list(genai.list_models())
            latency_ms = (time.time() - start_time) * 1000

            return DoctorResult(
                ok=True,
                message="Google AI API is accessible",
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
    """Register the Google provider with the global registry."""
    from llm_council.providers.registry import get_registry

    registry = get_registry()
    with contextlib.suppress(ValueError):
        registry.register_provider("google", GoogleProvider)


_register()
