"""
OpenAI provider adapter.

Direct integration with the OpenAI API for GPT models.
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

DEFAULT_MODEL = "gpt-5.1"

# Models that support structured output with json_schema response_format
# See: https://platform.openai.com/docs/guides/structured-outputs
STRUCTURED_OUTPUT_MODELS = frozenset(
    {
        # GPT-5.2 family (December 2025)
        "gpt-5.2",
        "gpt-5.2-codex",
        # GPT-5.1 family (2025)
        "gpt-5.1",
        "gpt-5.1-codex",
        "gpt-5.1-mini",
        "gpt-5.1-nano",
        # GPT-4o family (August 2024+)
        "gpt-4o",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-11-20",
        "gpt-4o-2025-01-15",
        # GPT-4o mini
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        # GPT-4.1 family
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        # o-series reasoning models
        "o1",
        "o1-2024-12-17",
        "o1-mini",
        "o1-mini-2024-09-12",
        "o3-mini",
        "o4-mini",
    }
)

# Model prefixes for easier matching (handles dated versions)
STRUCTURED_OUTPUT_MODEL_PREFIXES = (
    "gpt-5",  # All GPT-5.x models
    "gpt-4o",  # All GPT-4o variants
    "gpt-4.1",  # All GPT-4.1 variants
    "o1",  # o1 series
    "o3",  # o3 series
    "o4",  # o4 series
)

# Models that only support simple JSON mode (not full schema)
JSON_MODE_ONLY_MODELS = frozenset(
    {
        "gpt-4-turbo",
        "gpt-4-turbo-2024-04-09",
        "gpt-4-turbo-preview",
        "gpt-4-1106-preview",
        "gpt-4-0125-preview",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
    }
)

# o-series reasoning models that support reasoning_effort parameter
# See: https://platform.openai.com/docs/guides/reasoning
REASONING_MODELS = frozenset(
    {
        "o1",
        "o1-2024-12-17",
        "o1-mini",
        "o1-mini-2024-09-12",
        "o3",
        "o3-mini",
        "o3-pro",
        "o4-mini",
    }
)

# Prefixes for reasoning model detection
REASONING_MODEL_PREFIXES = ("o1", "o3", "o4")

# Models that use max_completion_tokens instead of max_tokens
# GPT-5.x and o-series use the new parameter
MAX_COMPLETION_TOKENS_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _make_schema_strict_compatible(schema: dict[str, Any]) -> dict[str, Any]:
    """Transform a JSON schema for OpenAI strict mode compatibility.

    OpenAI's strict mode requires:
    1. ALL properties must be listed in the `required` array
    2. ALL object types must have `additionalProperties: false`

    This function recursively processes the schema to:
    1. Remove $schema meta field (not needed by OpenAI)
    2. Add all properties to the required array
    3. Add additionalProperties: false to all objects
    4. Process nested object schemas recursively

    Args:
        schema: The original JSON schema.

    Returns:
        A new schema compatible with OpenAI strict mode.

    See: https://platform.openai.com/docs/guides/structured-outputs#additionalproperties
    """
    result: dict[str, Any] = {}

    for key, value in schema.items():
        # Skip $schema meta field
        if key == "$schema":
            continue
        # Skip additionalProperties - we'll set it ourselves
        if key == "additionalProperties":
            continue

        if key == "properties" and isinstance(value, dict):
            # Recursively process nested object properties
            result[key] = {
                prop_name: _make_schema_strict_compatible(prop_schema)
                if isinstance(prop_schema, dict) and prop_schema.get("type") == "object"
                else (
                    {
                        **prop_schema,
                        "items": _make_schema_strict_compatible(prop_schema["items"]),
                    }
                    if isinstance(prop_schema, dict)
                    and prop_schema.get("type") == "array"
                    and isinstance(prop_schema.get("items"), dict)
                    and prop_schema["items"].get("type") == "object"
                    else prop_schema
                )
                for prop_name, prop_schema in value.items()
            }
            # Make ALL properties required for strict mode
            result["required"] = list(value.keys())
            # Strict mode requires additionalProperties: false
            result["additionalProperties"] = False
        elif key == "required":
            # Skip - we'll set this when processing properties
            continue
        elif isinstance(value, dict) and value.get("type") == "object":
            # Recursively process nested object schemas
            result[key] = _make_schema_strict_compatible(value)
        else:
            result[key] = value

    # Ensure all object types have additionalProperties: false
    if schema.get("type") == "object" and "additionalProperties" not in result:
        result["additionalProperties"] = False

    return result


logger = logging.getLogger(__name__)


class OpenAIProvider(ProviderAdapter):
    """OpenAI API provider adapter.

    Direct access to GPT models via the OpenAI API.

    Environment variables:
        OPENAI_API_KEY: Required. Your OpenAI API key.

    Requires the 'openai' extra:
        pip install the-llm-council[openai]
    """

    name: ClassVar[str] = "openai"
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        streaming=True,
        tool_use=True,
        structured_output=True,
        multimodal=True,
        max_tokens=16384,
    )

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str | None = None,
    ) -> None:
        """Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            default_model: Default model to use if not specified in request.
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._default_model = default_model or DEFAULT_MODEL
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or create the OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError(
                    "The 'openai' package is required for the OpenAI provider. "
                    "Install it with: pip install the-llm-council[openai]"
                ) from e

            if not self._api_key:
                raise ValueError(
                    "OpenAI API key not configured. "
                    "Set OPENAI_API_KEY environment variable or pass api_key."
                )

            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def generate(
        self, request: GenerateRequest
    ) -> GenerateResponse | AsyncIterator[GenerateResponse]:
        """Generate a response using OpenAI API."""
        client = self._get_client()

        # Build messages
        messages = []
        if request.messages:
            for m in request.messages:
                messages.append({"role": m.role, "content": m.content})
        elif request.prompt:
            messages = [{"role": "user", "content": request.prompt}]
        else:
            raise ValueError("Either 'messages' or 'prompt' must be provided")

        kwargs: dict[str, Any] = {
            "model": request.model or self._default_model,
            "messages": messages,
        }

        if request.max_tokens is not None:
            model = request.model or self._default_model
            # GPT-5.x and o-series use max_completion_tokens instead of max_tokens
            if any(model.startswith(p) for p in MAX_COMPLETION_TOKENS_PREFIXES):
                kwargs["max_completion_tokens"] = request.max_tokens
            else:
                kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.stop:
            kwargs["stop"] = list(request.stop)
        if request.tools:
            kwargs["tools"] = list(request.tools)
        if request.tool_choice is not None:
            kwargs["tool_choice"] = request.tool_choice

        # Handle structured output - transform to OpenAI's native format
        if request.structured_output:
            model = request.model or self._default_model
            if self._model_supports_structured_output(model):
                # Transform schema for strict mode compatibility
                # OpenAI strict mode requires ALL properties in required array
                transformed_schema = _make_schema_strict_compatible(
                    dict(request.structured_output.json_schema)
                )
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": request.structured_output.name,
                        "strict": request.structured_output.strict,
                        "schema": transformed_schema,
                    },
                }
            elif model in JSON_MODE_ONLY_MODELS:
                # Fall back to simple JSON mode for older models
                kwargs["response_format"] = {"type": "json_object"}
            # else: model doesn't support structured output, skip
        elif request.response_format:
            # Legacy pass-through for backwards compatibility
            kwargs["response_format"] = dict(request.response_format)

        # Handle reasoning configuration for o-series models
        if request.reasoning and request.reasoning.enabled:
            model = request.model or self._default_model
            if self._model_supports_reasoning(model):
                # Default to "medium" if effort not specified
                effort = request.reasoning.effort or "medium"
                # Note: "none" is only valid for GPT-5.2+, o-series requires low/medium/high
                if effort == "none" and model.startswith(REASONING_MODEL_PREFIXES):
                    logger.warning(
                        "reasoning_effort='none' not supported for o-series model %s, using 'medium'",
                        model,
                    )
                    effort = "medium"
                kwargs["reasoning_effort"] = effort
            else:
                logger.warning(
                    "Model %s does not support reasoning_effort parameter; ignored",
                    model,
                )

        if request.stream:
            kwargs["stream"] = True
            return self._generate_stream(client, kwargs)

        response = await client.chat.completions.create(**kwargs)
        return self._parse_response(response)

    async def _generate_stream(
        self, client: Any, kwargs: dict[str, Any]
    ) -> AsyncIterator[GenerateResponse]:
        """Stream responses from OpenAI API."""
        stream = await client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield GenerateResponse(
                    text=chunk.choices[0].delta.content,
                    content=chunk.choices[0].delta.content,
                    finish_reason=chunk.choices[0].finish_reason,
                )

    def _parse_response(self, response: Any) -> GenerateResponse:
        """Parse OpenAI API response."""
        choice = response.choices[0]
        message = choice.message

        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return GenerateResponse(
            text=message.content,
            content=message.content,
            tool_calls=tool_calls,
            usage=usage,
            model=response.model,
            finish_reason=choice.finish_reason,
            raw=response,
        )

    def _model_supports_structured_output(self, model: str) -> bool:
        """Check if a specific model supports structured output with JSON schema.

        Supported models include:
        - GPT-5.x family (gpt-5.1, gpt-5.2, gpt-5.1-codex, gpt-5.2-codex)
        - GPT-4o family (gpt-4o, gpt-4o-mini)
        - GPT-4.1 family (gpt-4.1, gpt-4.1-mini, gpt-4.1-nano)
        - o-series reasoning models (o1, o3-mini, o4-mini)

        Args:
            model: The model identifier to check.

        Returns:
            True if the model supports json_schema response_format.
        """
        # Direct match
        if model in STRUCTURED_OUTPUT_MODELS:
            return True

        # Check if model starts with any supported prefix
        if any(model.startswith(prefix) for prefix in STRUCTURED_OUTPUT_MODEL_PREFIXES):
            return True

        # Check for version-less model names (e.g., "gpt-4o-2024-08-06" -> "gpt-4o")
        for suffix in ("-2024", "-2025", "-2026"):
            if suffix in model:
                base_model = model.split(suffix)[0]
                if base_model in STRUCTURED_OUTPUT_MODELS:
                    return True
                break

        return False

    def _model_supports_reasoning(self, model: str) -> bool:
        """Check if a model supports the reasoning_effort parameter.

        Only o-series reasoning models (o1, o3, o4-mini) support this parameter.

        Args:
            model: The model identifier to check.

        Returns:
            True if the model supports reasoning_effort parameter.
        """
        # Direct match
        if model in REASONING_MODELS:
            return True

        # Check if model starts with any o-series prefix
        return any(model.startswith(prefix) for prefix in REASONING_MODEL_PREFIXES)

    async def supports(self, capability: str) -> bool:
        """Check if the provider supports a capability."""
        if not self.supports_capability_name(capability):
            return False
        return getattr(self.capabilities, capability, False)

    async def doctor(self) -> DoctorResult:
        """Perform a health check on the OpenAI API."""
        start_time = time.time()

        if not self._api_key:
            return DoctorResult(
                ok=False,
                message="OPENAI_API_KEY environment variable not set",
                details={"error": "missing_api_key"},
            )

        try:
            from openai import AsyncOpenAI
        except ImportError:
            return DoctorResult(
                ok=False,
                message="openai package not installed. Run: pip install the-llm-council[openai]",
                details={"error": "missing_package"},
            )

        try:
            client = self._get_client()
            await client.models.list()
            latency_ms = (time.time() - start_time) * 1000

            return DoctorResult(
                ok=True,
                message="OpenAI API is accessible",
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
    """Register the OpenAI provider with the global registry."""
    from llm_council.providers.registry import get_registry

    registry = get_registry()
    with contextlib.suppress(ValueError):
        registry.register_provider("openai", OpenAIProvider)


_register()
