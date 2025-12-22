"""
Anthropic provider adapter.

Direct integration with the Anthropic API for Claude models.
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

DEFAULT_MODEL = "claude-3-5-sonnet-20241022"


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

        if request.stream:
            return self._generate_stream(client, kwargs)

        response = await client.messages.create(**kwargs)
        return self._parse_response(response)

    async def _generate_stream(
        self, client: Any, kwargs: dict[str, Any]
    ) -> AsyncIterator[GenerateResponse]:
        """Stream responses from Anthropic API."""
        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield GenerateResponse(text=text, content=text)

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
