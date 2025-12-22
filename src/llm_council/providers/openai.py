"""
OpenAI provider adapter.

Direct integration with the OpenAI API for GPT models.
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

DEFAULT_MODEL = "gpt-4o"


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
        if request.response_format:
            kwargs["response_format"] = dict(request.response_format)

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
