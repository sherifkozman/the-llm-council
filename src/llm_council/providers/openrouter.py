"""
OpenRouter provider adapter.

OpenRouter provides unified API access to multiple LLM providers
with a single API key, making it the recommended default for LLM Council.

Docs: https://openrouter.ai/docs
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import httpx

from llm_council.providers.base import (
    DoctorResult,
    GenerateRequest,
    GenerateResponse,
    ProviderAdapter,
    ProviderCapabilities,
)

# Default models for different use cases (December 2025)
DEFAULT_MODEL = "anthropic/claude-opus-4-5"
FAST_MODEL = "anthropic/claude-3-5-haiku"
REASONING_MODEL = "anthropic/claude-opus-4-5"
CODE_MODEL = "openai/gpt-5.1"
CRITIC_MODEL = "anthropic/claude-sonnet-4-5"


def _make_schema_strict_compatible(schema: dict[str, Any]) -> dict[str, Any]:
    """Transform a JSON schema for OpenAI strict mode compatibility.

    OpenRouter uses OpenAI-compatible format, which in strict mode requires:
    1. ALL properties must be listed in the `required` array
    2. ALL object types must have `additionalProperties: false`

    This function recursively processes the schema to:
    1. Remove $schema meta field (not needed)
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


class OpenRouterProvider(ProviderAdapter):
    """OpenRouter API provider adapter.

    OpenRouter provides access to 100+ models through a unified API.
    This is the recommended default provider for LLM Council due to:
    - Single API key for all models
    - Automatic failover and routing
    - Competitive pricing

    Environment variables:
        OPENROUTER_API_KEY: Required. Your OpenRouter API key.
        OPENROUTER_BASE_URL: Optional. Override the API base URL.
        OPENROUTER_HTTP_REFERER: Optional. Custom HTTP-Referer header.
        OPENROUTER_APP_TITLE: Optional. Custom X-Title header.
    """

    name: ClassVar[str] = "openrouter"
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        streaming=True,
        tool_use=True,
        structured_output=True,
        multimodal=True,
        max_tokens=None,  # Varies by model
    )

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the OpenRouter provider.

        Args:
            api_key: OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
            base_url: Override the API base URL.
            default_model: Default model to use if not specified in request.
            http_client: Optional custom HTTP client for testing.
        """
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._base_url = base_url or os.environ.get("OPENROUTER_BASE_URL", self.BASE_URL)
        self._default_model = default_model or DEFAULT_MODEL
        self._http_client = http_client
        self._owns_client = http_client is None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        return self._http_client

    async def _close_client(self) -> None:
        """Close the HTTP client if we own it."""
        if self._http_client is not None and self._owns_client:
            await self._http_client.aclose()
            self._http_client = None

    def _get_headers(self) -> dict[str, str]:
        """Build request headers.

        Headers HTTP-Referer and X-Title can be customized via environment
        variables for third-party integrations.
        """
        if not self._api_key:
            raise ValueError(
                "OpenRouter API key not configured. "
                "Set OPENROUTER_API_KEY environment variable or pass api_key."
            )
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get(
                "OPENROUTER_HTTP_REFERER", "https://github.com/sherifkozman/the-llm-council"
            ),
            "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "LLM Council"),
        }

    def _build_request_body(self, request: GenerateRequest) -> dict[str, Any]:
        """Convert GenerateRequest to OpenRouter API format."""
        body: dict[str, Any] = {
            "model": request.model or self._default_model,
        }

        # Handle messages vs prompt
        if request.messages:
            body["messages"] = [
                {"role": m.role, "content": m.content, **({"name": m.name} if m.name else {})}
                for m in request.messages
            ]
        elif request.prompt:
            body["messages"] = [{"role": "user", "content": request.prompt}]
        else:
            raise ValueError("Either 'messages' or 'prompt' must be provided")

        # Optional parameters
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.stop:
            body["stop"] = list(request.stop)
        if request.stream:
            body["stream"] = True
        if request.tools:
            body["tools"] = list(request.tools)
        if request.tool_choice is not None:
            body["tool_choice"] = request.tool_choice

        # Handle structured output - transform to OpenRouter's format (OpenAI-compatible)
        # See: https://openrouter.ai/docs/guides/features/structured-outputs
        #
        # Note: OpenRouter internally handles model compatibility. If the underlying
        # model doesn't support json_schema, OpenRouter may:
        # - Return an error
        # - Fall back to prompt-based JSON
        # - Use its Response Healing feature
        #
        # We apply the format for all models and let OpenRouter handle compatibility.
        if request.structured_output:
            # Transform schema for strict mode compatibility
            # OpenAI strict mode requires ALL properties in required array
            transformed_schema = _make_schema_strict_compatible(
                dict(request.structured_output.json_schema)
            )
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.structured_output.name,
                    "strict": request.structured_output.strict,
                    "schema": transformed_schema,
                },
            }
        elif request.response_format:
            # Legacy pass-through for backwards compatibility
            body["response_format"] = dict(request.response_format)

        # Handle reasoning configuration
        # OpenRouter passes through to underlying models (OpenAI o-series, etc.)
        if request.reasoning and request.reasoning.enabled:
            if request.reasoning.effort:
                # Pass through reasoning_effort for OpenAI o-series models
                body["reasoning_effort"] = request.reasoning.effort

        return body

    async def generate(
        self, request: GenerateRequest
    ) -> GenerateResponse | AsyncIterator[GenerateResponse]:
        """Generate a response using OpenRouter API.

        Args:
            request: The generation request.

        Returns:
            GenerateResponse for non-streaming, or async iterator for streaming.

        Raises:
            ValueError: If request is invalid.
            httpx.HTTPError: If API call fails.
        """
        client = await self._get_client()
        body = self._build_request_body(request)

        if request.stream:
            return self._generate_stream(client, body)

        response = await client.post(
            f"{self._base_url}/chat/completions",
            headers=self._get_headers(),
            json=body,
        )
        response.raise_for_status()
        data = response.json()

        return self._parse_response(data)

    async def _generate_stream(
        self, client: httpx.AsyncClient, body: dict[str, Any]
    ) -> AsyncIterator[GenerateResponse]:
        """Stream responses from OpenRouter API."""
        async with client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            headers=self._get_headers(),
            json=body,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]  # Remove "data: " prefix
                if data_str == "[DONE]":
                    break
                try:
                    import json

                    data = json.loads(data_str)
                    yield self._parse_stream_chunk(data)
                except Exception:
                    continue

    def _parse_response(self, data: dict[str, Any]) -> GenerateResponse:
        """Parse OpenRouter API response."""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        usage = data.get("usage", {})
        usage_dict = None
        if usage:
            usage_dict = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        return GenerateResponse(
            text=message.get("content"),
            content=message.get("content"),
            tool_calls=message.get("tool_calls"),
            usage=usage_dict,
            model=data.get("model"),
            finish_reason=choice.get("finish_reason"),
            raw=data,
        )

    def _parse_stream_chunk(self, data: dict[str, Any]) -> GenerateResponse:
        """Parse a streaming chunk from OpenRouter API."""
        choice = data.get("choices", [{}])[0]
        delta = choice.get("delta", {})

        return GenerateResponse(
            text=delta.get("content"),
            content=delta.get("content"),
            tool_calls=delta.get("tool_calls"),
            finish_reason=choice.get("finish_reason"),
            raw=data,
        )

    async def supports(self, capability: str) -> bool:
        """Check if the provider supports a capability.

        Args:
            capability: Capability name (e.g., 'streaming', 'tool_use').

        Returns:
            True if the capability is supported.
        """
        if not self.supports_capability_name(capability):
            return False
        return getattr(self.capabilities, capability, False)

    async def doctor(self) -> DoctorResult:
        """Perform a health check on the OpenRouter API.

        Returns:
            DoctorResult with status and diagnostics.
        """
        start_time = time.time()

        # Check for API key
        if not self._api_key:
            return DoctorResult(
                ok=False,
                message="OPENROUTER_API_KEY environment variable not set",
                details={"error": "missing_api_key"},
            )

        try:
            client = await self._get_client()

            # Make a minimal API call to verify connectivity
            response = await client.get(
                f"{self._base_url}/models",
                headers=self._get_headers(),
            )
            response.raise_for_status()

            latency_ms = (time.time() - start_time) * 1000

            return DoctorResult(
                ok=True,
                message="OpenRouter API is accessible",
                latency_ms=latency_ms,
                details={"models_available": True},
            )

        except httpx.HTTPStatusError as e:
            latency_ms = (time.time() - start_time) * 1000
            return DoctorResult(
                ok=False,
                message=f"API error: {e.response.status_code}",
                latency_ms=latency_ms,
                details={"status_code": e.response.status_code},
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return DoctorResult(
                ok=False,
                message=f"Connection error: {str(e)}",
                latency_ms=latency_ms,
                details={"error": str(e)},
            )


def create_openrouter_for_model(model: str) -> OpenRouterProvider:
    """Create an OpenRouter provider instance configured for a specific model.

    This factory function is used to create multiple provider instances
    for multi-model council runs.

    Args:
        model: OpenRouter model ID (e.g., "anthropic/claude-3.5-sonnet").

    Returns:
        Configured OpenRouterProvider instance.

    Example:
        >>> provider = create_openrouter_for_model("openai/gpt-4o")
        >>> # This provider will always use gpt-4o for requests
    """
    return OpenRouterProvider(default_model=model)


# Register the provider
def _register() -> None:
    """Register the OpenRouter provider with the global registry."""
    from llm_council.providers.registry import get_registry

    registry = get_registry()
    try:
        registry.register_provider("openrouter", OpenRouterProvider)
    except ValueError:
        pass  # Already registered


_register()
