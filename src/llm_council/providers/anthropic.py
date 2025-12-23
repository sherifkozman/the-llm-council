"""
Anthropic provider adapter.

Direct integration with the Anthropic API for Claude models.
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

DEFAULT_MODEL = "claude-opus-4-5"

# Beta header required for structured outputs
# See: https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs
STRUCTURED_OUTPUTS_BETA = "structured-outputs-2025-11-13"

# Model prefixes that support structured output with output_format
# Requires beta header: anthropic-beta: structured-outputs-2025-11-13
# Claude 4.x family (December 2025+) supports structured outputs
# Claude 3.x family does NOT support structured outputs
STRUCTURED_OUTPUT_MODEL_PREFIXES = (
    "claude-opus-4",  # Claude Opus 4.x series
    "claude-sonnet-4",  # Claude Sonnet 4.x series
    "claude-haiku-4",  # Claude Haiku 4.x series
    "claude-4",  # Alternative naming pattern
)

# Explicit model names known to support structured output
STRUCTURED_OUTPUT_MODELS = frozenset(
    {
        # Claude Opus 4.x
        "claude-opus-4-5",
        "claude-opus-4-1",
        # Claude Sonnet 4.x
        "claude-sonnet-4-5",
        # Claude Haiku 4.x
        "claude-haiku-4-5",
    }
)


def _strip_schema_meta_fields(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip meta fields from JSON schema that Anthropic's API doesn't need.

    Removes `$schema` and other JSON Schema meta fields that aren't part
    of the actual schema definition.

    Args:
        schema: The original JSON schema.

    Returns:
        A new schema without meta fields.
    """
    result: dict[str, Any] = {}

    for key, value in schema.items():
        # Skip $schema and other meta fields
        if key in ("$schema", "$id", "$ref", "$comment"):
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


logger = logging.getLogger(__name__)


class AnthropicProvider(ProviderAdapter):
    """Anthropic API provider adapter.

    Direct access to Claude models via the Anthropic API.

    Environment variables:
        ANTHROPIC_API_KEY: Required. Your Anthropic API key.

    Requires the 'anthropic' extra:
        pip install the-llm-council[anthropic]
    """

    name: ClassVar[str] = "anthropic"
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
        """Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
            default_model: Default model to use if not specified in request.
        """
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._default_model = default_model or DEFAULT_MODEL
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or create the Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as e:
                raise ImportError(
                    "The 'anthropic' package is required for the Anthropic provider. "
                    "Install it with: pip install the-llm-council[anthropic]"
                ) from e

            if not self._api_key:
                raise ValueError(
                    "Anthropic API key not configured. "
                    "Set ANTHROPIC_API_KEY environment variable or pass api_key."
                )

            self._client = AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def generate(
        self, request: GenerateRequest
    ) -> GenerateResponse | AsyncIterator[GenerateResponse]:
        """Generate a response using Anthropic API."""
        client = self._get_client()

        # Build messages
        messages = []
        system = None
        if request.messages:
            for m in request.messages:
                if m.role == "system":
                    system = m.content
                else:
                    messages.append({"role": m.role, "content": m.content})
        elif request.prompt:
            messages = [{"role": "user", "content": request.prompt}]
        else:
            raise ValueError("Either 'messages' or 'prompt' must be provided")

        kwargs: dict[str, Any] = {
            "model": request.model or self._default_model,
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
        }

        if system:
            kwargs["system"] = system
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.stop:
            kwargs["stop_sequences"] = list(request.stop)
        if request.tools:
            kwargs["tools"] = list(request.tools)

        # Handle structured output - requires beta header and output_format
        # See: https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs
        use_beta = False
        model = request.model or self._default_model

        if request.structured_output:
            if self._model_supports_structured_output(model):
                use_beta = True
                # Strip $schema and other meta fields
                kwargs["output_format"] = {
                    "type": "json_schema",
                    "schema": _strip_schema_meta_fields(
                        dict(request.structured_output.json_schema)
                    ),
                }
            # else: model doesn't support structured output, skip (rely on prompt)
        elif request.response_format:
            # Legacy response_format: attempt conversion if model supports structured output
            # and format looks like OpenAI-style json_schema
            if (
                self._model_supports_structured_output(model)
                and isinstance(request.response_format, dict)
                and request.response_format.get("type") == "json_schema"
            ):
                use_beta = True
                # Handle both OpenAI format (nested json_schema) and flat format
                json_schema = request.response_format.get("json_schema", {})
                schema = (
                    json_schema.get("schema")
                    if json_schema
                    else request.response_format.get("schema")
                )
                if schema:
                    kwargs["output_format"] = {
                        "type": "json_schema",
                        "schema": dict(schema),
                    }
            # Note: simple json_object mode is not supported by Anthropic API

        # Handle reasoning/thinking configuration
        # See: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
        if request.reasoning and request.reasoning.enabled:
            use_beta = True  # Extended thinking requires beta API
            # Anthropic uses budget_tokens with min 1024, max 128000
            budget = request.reasoning.budget_tokens or 8192
            budget = max(min(budget, 128000), 1024)  # Clamp to valid range
            if request.reasoning.budget_tokens and request.reasoning.budget_tokens != budget:
                logger.warning(
                    "Anthropic budget_tokens clamped from %d to %d (valid range: 1024-128000)",
                    request.reasoning.budget_tokens,
                    budget,
                )
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget,
            }

        if request.stream:
            return self._generate_stream(client, kwargs, use_beta=use_beta)

        if use_beta:
            response = await client.beta.messages.create(
                betas=[STRUCTURED_OUTPUTS_BETA],
                **kwargs,
            )
        else:
            response = await client.messages.create(**kwargs)
        return self._parse_response(response)

    async def _generate_stream(
        self, client: Any, kwargs: dict[str, Any], *, use_beta: bool = False
    ) -> AsyncIterator[GenerateResponse]:
        """Stream responses from Anthropic API."""
        if use_beta:
            async with client.beta.messages.stream(
                betas=[STRUCTURED_OUTPUTS_BETA], **kwargs
            ) as stream:
                async for text in stream.text_stream:
                    yield GenerateResponse(text=text, content=text)
        else:
            async with client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield GenerateResponse(text=text, content=text)

    def _model_supports_structured_output(self, model: str) -> bool:
        """Check if a specific model supports structured output.

        Structured outputs are supported on Claude 4.x family models:
        - Claude Opus 4.x (claude-opus-4-5, claude-opus-4-1, etc.)
        - Claude Sonnet 4.x (claude-sonnet-4-5, etc.)
        - Claude Haiku 4.x (claude-haiku-4-5, etc.)

        Claude 3.x models do NOT support structured outputs.

        Args:
            model: The model identifier to check.

        Returns:
            True if the model supports output_format with json_schema.
        """
        # Direct match against known models
        if model in STRUCTURED_OUTPUT_MODELS:
            return True

        # Strip date suffix for comparison (e.g., "claude-sonnet-4-5-20251201" -> "claude-sonnet-4-5")
        base_model = model
        for suffix in ("-2024", "-2025", "-2026"):
            if suffix in model:
                base_model = model.split(suffix)[0]
                break

        # Check if base model is in the known set
        if base_model in STRUCTURED_OUTPUT_MODELS:
            return True

        # Check if model starts with any supported prefix (Claude 4.x family)
        return any(model.startswith(prefix) for prefix in STRUCTURED_OUTPUT_MODEL_PREFIXES)

    def _parse_response(self, response: Any) -> GenerateResponse:
        """Parse Anthropic API response."""
        text = ""
        tool_calls = None

        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
            elif hasattr(block, "type") and block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {"name": block.name, "arguments": block.input},
                    }
                )

        usage = None
        if hasattr(response, "usage"):
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

        return GenerateResponse(
            text=text,
            content=text,
            tool_calls=tool_calls,
            usage=usage,
            model=response.model,
            finish_reason=response.stop_reason,
            raw=response,
        )

    async def supports(self, capability: str) -> bool:
        """Check if the provider supports a capability."""
        if not self.supports_capability_name(capability):
            return False
        return getattr(self.capabilities, capability, False)

    async def doctor(self) -> DoctorResult:
        """Perform a health check on the Anthropic API."""
        start_time = time.time()

        if not self._api_key:
            return DoctorResult(
                ok=False,
                message="ANTHROPIC_API_KEY environment variable not set",
                details={"error": "missing_api_key"},
            )

        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            return DoctorResult(
                ok=False,
                message="anthropic package not installed. Run: pip install the-llm-council[anthropic]",
                details={"error": "missing_package"},
            )

        try:
            client = self._get_client()
            # Minimal API call
            await client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1,
                messages=[{"role": "user", "content": "Hi"}],
            )
            latency_ms = (time.time() - start_time) * 1000

            return DoctorResult(
                ok=True,
                message="Anthropic API is accessible",
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
    """Register the Anthropic provider with the global registry."""
    from llm_council.providers.registry import get_registry

    registry = get_registry()
    with contextlib.suppress(ValueError):
        registry.register_provider("anthropic", AnthropicProvider)


_register()
