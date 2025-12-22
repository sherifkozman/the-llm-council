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

DEFAULT_MODEL = "gemini-2.0-flash-exp"


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
