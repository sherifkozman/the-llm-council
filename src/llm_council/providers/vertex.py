"""
Vertex AI provider adapter.

Enterprise access to Google Cloud's AI models through Vertex AI.
Uses the google-genai SDK with Vertex AI backend for unified API access.

Supports both Application Default Credentials (ADC) and service account authentication.
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

# Import shared schema stripping logic from google provider
from llm_council.providers.google import (
    LEGACY_MODEL_PREFIXES,
    STRUCTURED_OUTPUT_MODEL_PREFIXES,
    _strip_schema_meta_fields,
)

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_LOCATION = "us-central1"

logger = logging.getLogger(__name__)


class VertexAIProvider(ProviderAdapter):
    """Vertex AI provider adapter.

    Enterprise access to Gemini and other models via Google Cloud's Vertex AI platform.
    Uses the google-genai SDK with Vertex AI backend.

    Authentication (in order of priority):
        1. Explicit credentials passed to constructor
        2. Service account JSON file via GOOGLE_APPLICATION_CREDENTIALS env var
        3. Application Default Credentials (ADC) via gcloud CLI

    Environment variables:
        GOOGLE_CLOUD_PROJECT: Required. Your GCP project ID.
        GOOGLE_CLOUD_LOCATION: Optional. Region (default: us-central1).
        GOOGLE_APPLICATION_CREDENTIALS: Optional. Path to service account JSON.

    Requires the 'vertex' extra:
        pip install the-llm-council[vertex]

    Usage:
        # Using ADC (run 'gcloud auth application-default login' first)
        export GOOGLE_CLOUD_PROJECT=my-project
        council run architect "Design a cache" --providers vertex-ai

        # Using service account
        export GOOGLE_CLOUD_PROJECT=my-project
        export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
        council run architect "Design a cache" --providers vertex-ai
    """

    name: ClassVar[str] = "vertex-ai"
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        streaming=True,
        tool_use=True,
        structured_output=True,
        multimodal=True,
        max_tokens=8192,
    )

    def __init__(
        self,
        project: str | None = None,
        location: str | None = None,
        credentials: Any = None,
        default_model: str | None = None,
    ) -> None:
        """Initialize the Vertex AI provider.

        Args:
            project: GCP project ID. Falls back to GOOGLE_CLOUD_PROJECT env var.
            location: GCP region. Falls back to GOOGLE_CLOUD_LOCATION or 'us-central1'.
            credentials: Optional google.auth.credentials.Credentials object.
                        If not provided, uses ADC or GOOGLE_APPLICATION_CREDENTIALS.
            default_model: Default model to use if not specified in request.
        """
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", DEFAULT_LOCATION)
        self._credentials = credentials
        self._default_model = default_model or DEFAULT_MODEL
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or create the Vertex AI GenAI client."""
        if self._client is None:
            try:
                from google import genai
            except ImportError as e:
                raise ImportError(
                    "The 'google-genai' package is required for the Vertex AI provider. "
                    "Install it with: pip install the-llm-council[vertex]"
                ) from e

            if not self._project:
                raise ValueError(
                    "Google Cloud project not configured. "
                    "Set GOOGLE_CLOUD_PROJECT environment variable or pass project parameter."
                )

            # Load credentials if not provided
            credentials = self._credentials
            if credentials is None:
                credentials = self._load_credentials()

            # Create client with Vertex AI backend
            self._client = genai.Client(
                vertexai=True,
                project=self._project,
                location=self._location,
                credentials=credentials,
            )

        return self._client

    def _load_credentials(self) -> Any:
        """Load credentials from environment or ADC.

        The google-genai SDK handles credential loading automatically:
        1. Checks GOOGLE_APPLICATION_CREDENTIALS (service account or ADC file)
        2. Falls back to Application Default Credentials (ADC)

        We only need to explicitly load credentials if we want to use
        a specific service account file that's NOT in the standard location.

        Returns:
            google.auth.credentials.Credentials or None (let SDK handle it)
        """
        # Let the SDK handle credential loading automatically
        # It supports both service account files and ADC
        return None

    async def generate(
        self, request: GenerateRequest
    ) -> GenerateResponse | AsyncIterator[GenerateResponse]:
        """Generate a response using Vertex AI."""
        client = self._get_client()
        model = request.model or self._default_model

        # Build content - same format as GoogleProvider
        contents: str | list[dict[str, Any]]
        if request.messages:
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

        # Handle structured output
        if request.structured_output:
            if self._model_supports_structured_output(model):
                config["response_mime_type"] = "application/json"
                config["response_schema"] = _strip_schema_meta_fields(
                    dict(request.structured_output.json_schema)
                )
            elif self._is_legacy_model(model):
                config["response_mime_type"] = "application/json"

        # Handle reasoning/thinking configuration
        if request.reasoning and request.reasoning.enabled:
            if request.reasoning.thinking_level:
                config["thinking_config"] = {
                    "thinking_level": request.reasoning.thinking_level.upper(),
                }
            elif request.reasoning.budget_tokens:
                max_budget = 24576
                budget = min(request.reasoning.budget_tokens, max_budget)
                if request.reasoning.budget_tokens > max_budget:
                    logger.warning(
                        "Vertex AI thinking_budget capped from %d to %d (provider maximum)",
                        request.reasoning.budget_tokens,
                        max_budget,
                    )
                config["thinking_config"] = {
                    "thinking_budget": budget,
                }
            else:
                config["thinking_config"] = {
                    "thinking_level": "MEDIUM",
                }

        if request.stream:
            return self._generate_stream(client, model, contents, config)

        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config if config else None,
        )
        return self._parse_response(response)

    async def _generate_stream(
        self, client: Any, model: str, contents: Any, config: dict[str, Any]
    ) -> AsyncIterator[GenerateResponse]:
        """Stream responses from Vertex AI."""
        stream = await client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config if config else None,
        )
        async for chunk in stream:
            if chunk.text:
                yield GenerateResponse(text=chunk.text, content=chunk.text)

    def _parse_response(self, response: Any) -> GenerateResponse:
        """Parse Vertex AI response."""
        text = ""
        try:
            text = response.text
        except Exception:
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
                "prompt_tokens": getattr(um, "prompt_token_count", 0),
                "completion_tokens": getattr(um, "candidates_token_count", 0),
                "total_tokens": getattr(um, "total_token_count", 0),
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
        """Check if a model supports structured output with response_schema."""
        return any(model.startswith(prefix) for prefix in STRUCTURED_OUTPUT_MODEL_PREFIXES)

    def _is_legacy_model(self, model: str) -> bool:
        """Check if a model only supports simple JSON mode (no schema)."""
        return any(model.startswith(prefix) for prefix in LEGACY_MODEL_PREFIXES)

    async def supports(self, capability: str) -> bool:
        """Check if the provider supports a capability."""
        if not self.supports_capability_name(capability):
            return False
        return getattr(self.capabilities, capability, False)

    async def doctor(self) -> DoctorResult:
        """Perform a health check on Vertex AI."""
        start_time = time.time()

        if not self._project:
            return DoctorResult(
                ok=False,
                message="GOOGLE_CLOUD_PROJECT environment variable not set",
                details={"error": "missing_project"},
            )

        try:
            from google import genai
        except ImportError:
            return DoctorResult(
                ok=False,
                message="google-genai package not installed. Run: pip install the-llm-council[vertex]",
                details={"error": "missing_package"},
            )

        try:
            client = self._get_client()
            # List models to verify credentials work
            list(client.models.list())
            latency_ms = (time.time() - start_time) * 1000

            return DoctorResult(
                ok=True,
                message=f"Vertex AI is accessible (project: {self._project}, location: {self._location})",
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = str(e)

            # Provide helpful error messages
            if "credentials" in error_msg.lower() or "authentication" in error_msg.lower():
                help_msg = (
                    "Authentication failed. Either:\n"
                    "  1. Run 'gcloud auth application-default login', or\n"
                    "  2. Set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON file"
                )
            elif "permission" in error_msg.lower():
                help_msg = f"Permission denied. Ensure the credentials have Vertex AI access in project '{self._project}'"
            else:
                help_msg = error_msg

            return DoctorResult(
                ok=False,
                message=f"API error: {help_msg}",
                latency_ms=latency_ms,
                details={"error": error_msg},
            )


def _register() -> None:
    """Register the Vertex AI provider with the global registry."""
    from llm_council.providers.registry import get_registry

    registry = get_registry()
    with contextlib.suppress(ValueError):
        registry.register_provider("vertex-ai", VertexAIProvider)


_register()
