"""
Vertex AI provider adapter.

Enterprise access to Google Cloud's AI models through Vertex AI.
Supports both Gemini and Claude models with automatic SDK routing.

Gemini models: Uses google-genai SDK with Vertex AI backend.
Claude models: Uses anthropic[vertex] SDK.

Supports both Application Default Credentials (ADC) and service account authentication.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any, ClassVar

from llm_council.providers.anthropic import (
    STRUCTURED_OUTPUT_MODEL_PREFIXES as ANTHROPIC_STRUCTURED_OUTPUT_PREFIXES,
)
from llm_council.providers.anthropic import (
    STRUCTURED_OUTPUT_MODELS as ANTHROPIC_STRUCTURED_OUTPUT_MODELS,
)
from llm_council.providers.anthropic import (
    STRUCTURED_OUTPUTS_BETA,
)

# Import shared schema stripping from anthropic provider
from llm_council.providers.anthropic import (
    _strip_schema_meta_fields as _strip_anthropic_schema_fields,
)
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
DEFAULT_CLAUDE_REGION = "global"

# Environment variable names
ENV_VERTEX_MODEL = "VERTEX_AI_MODEL"
ENV_ANTHROPIC_MODEL = "ANTHROPIC_MODEL"
ENV_ANTHROPIC_PROJECT = "ANTHROPIC_VERTEX_PROJECT_ID"
ENV_CLAUDE_REGION = "CLOUD_ML_REGION"

# Model prefixes for detection
CLAUDE_MODEL_PREFIXES = ("claude-",)

logger = logging.getLogger(__name__)


class VertexAIProvider(ProviderAdapter):
    """Vertex AI provider adapter.

    Enterprise access to Gemini and Claude models via Google Cloud's Vertex AI platform.
    Automatically routes to the appropriate SDK based on model type:
    - Gemini models: Uses google-genai SDK (region: us-central1)
    - Claude models: Uses anthropic[vertex] SDK (region: global)

    Authentication (in order of priority):
        1. Explicit credentials passed to constructor
        2. Service account JSON file via GOOGLE_APPLICATION_CREDENTIALS env var
        3. Application Default Credentials (ADC) via gcloud CLI

    Environment variables for Gemini:
        GOOGLE_CLOUD_PROJECT: Required. Your GCP project ID.
        GOOGLE_CLOUD_LOCATION: Optional. Region (default: us-central1).
        VERTEX_AI_MODEL: Optional. Default Gemini model ID.

    Environment variables for Claude:
        ANTHROPIC_VERTEX_PROJECT_ID: Required for Claude. Your GCP project ID.
        CLOUD_ML_REGION: Optional. Region for Claude (default: global).
        ANTHROPIC_MODEL: Optional. Default Claude model ID (e.g., claude-opus-4-5@20251101).

    Requires the 'vertex' extra for Gemini:
        pip install the-llm-council[vertex]

    Requires anthropic[vertex] for Claude:
        pip install anthropic[vertex]

    Usage:
        # Gemini models
        export GOOGLE_CLOUD_PROJECT=my-project
        export VERTEX_AI_MODEL=gemini-2.5-pro
        council run architect "Design a cache" --providers vertex-ai

        # Claude models
        export ANTHROPIC_VERTEX_PROJECT_ID=my-project
        export ANTHROPIC_MODEL=claude-opus-4-5@20251101
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
        # Gemini configuration
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", DEFAULT_LOCATION)
        self._credentials = credentials

        # Claude configuration (separate project/region)
        self._claude_project = os.environ.get(ENV_ANTHROPIC_PROJECT)
        self._claude_region = os.environ.get(ENV_CLAUDE_REGION, DEFAULT_CLAUDE_REGION)

        # Determine default model - check ANTHROPIC_MODEL first for Claude, then VERTEX_AI_MODEL for Gemini
        anthropic_model = os.environ.get(ENV_ANTHROPIC_MODEL)
        vertex_model = os.environ.get(ENV_VERTEX_MODEL)

        if default_model:
            self._default_model = default_model
        elif anthropic_model:
            # Claude model configured
            self._default_model = anthropic_model
        elif vertex_model:
            # Gemini model configured
            self._default_model = vertex_model
        else:
            self._default_model = DEFAULT_MODEL

        # Clients for each model type
        self._gemini_client: Any = None
        self._claude_client: Any = None

    def _is_claude_model(self, model: str) -> bool:
        """Check if a model is a Claude model."""
        return any(model.startswith(prefix) for prefix in CLAUDE_MODEL_PREFIXES)

    def _get_gemini_client(self) -> Any:
        """Get or create the Gemini Vertex AI client."""
        if self._gemini_client is None:
            try:
                from google import genai
            except ImportError as e:
                raise ImportError(
                    "The 'google-genai' package is required for Gemini on Vertex AI. "
                    "Install it with: pip install the-llm-council[vertex]"
                ) from e

            if not self._project:
                raise ValueError(
                    "Google Cloud project not configured for Gemini. "
                    "Set GOOGLE_CLOUD_PROJECT environment variable or pass project parameter."
                )

            # Load credentials if not provided
            credentials = self._credentials
            if credentials is None:
                credentials = self._load_credentials()

            # Create client with Vertex AI backend
            self._gemini_client = genai.Client(
                vertexai=True,
                project=self._project,
                location=self._location,
                credentials=credentials,
            )

        return self._gemini_client

    def _get_claude_client(self) -> Any:
        """Get or create the Claude Vertex AI client."""
        if self._claude_client is None:
            try:
                from anthropic import AsyncAnthropicVertex
            except ImportError as e:
                raise ImportError(
                    "The 'anthropic[vertex]' package is required for Claude on Vertex AI. "
                    "Install it with: pip install anthropic[vertex]"
                ) from e

            if not self._claude_project:
                raise ValueError(
                    "GCP project not configured for Claude on Vertex AI. "
                    "Set ANTHROPIC_VERTEX_PROJECT_ID environment variable."
                )

            self._claude_client = AsyncAnthropicVertex(
                project_id=self._claude_project,
                region=self._claude_region,
            )

        return self._claude_client

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
        """Generate a response using Vertex AI.

        Routes to the appropriate SDK based on model type:
        - Claude models: Uses anthropic[vertex] SDK
        - Gemini models: Uses google-genai SDK
        """
        model = request.model or self._default_model

        # Route to appropriate implementation based on model type
        if self._is_claude_model(model):
            return await self._generate_claude(request, model)
        else:
            return await self._generate_gemini(request, model)

    async def _generate_gemini(
        self, request: GenerateRequest, model: str
    ) -> GenerateResponse | AsyncIterator[GenerateResponse]:
        """Generate response using Gemini via google-genai SDK."""
        client = self._get_gemini_client()

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

    async def _generate_claude(
        self, request: GenerateRequest, model: str
    ) -> GenerateResponse | AsyncIterator[GenerateResponse]:
        """Generate response using Claude via anthropic[vertex] SDK."""
        client = self._get_claude_client()

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
            "model": model,
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

        # Handle structured output for Claude 4.x models
        use_beta = False
        if request.structured_output:
            if self._claude_model_supports_structured_output(model):
                use_beta = True
                kwargs["output_format"] = {
                    "type": "json_schema",
                    "schema": _strip_anthropic_schema_fields(
                        dict(request.structured_output.json_schema)
                    ),
                }

        # Handle reasoning/thinking for Claude
        if request.reasoning and request.reasoning.enabled:
            use_beta = True
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
            return self._generate_claude_stream(client, kwargs, use_beta=use_beta)

        if use_beta:
            response = await client.beta.messages.create(
                betas=[STRUCTURED_OUTPUTS_BETA],
                **kwargs,
            )
        else:
            response = await client.messages.create(**kwargs)
        return self._parse_claude_response(response)

    async def _generate_claude_stream(
        self, client: Any, kwargs: dict[str, Any], *, use_beta: bool = False
    ) -> AsyncIterator[GenerateResponse]:
        """Stream responses from Claude on Vertex AI."""
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

    def _parse_claude_response(self, response: Any) -> GenerateResponse:
        """Parse Claude API response."""
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

    def _claude_model_supports_structured_output(self, model: str) -> bool:
        """Check if a Claude model supports structured output."""
        # Strip version suffix for comparison (e.g., "claude-opus-4-5@20251101" -> "claude-opus-4-5")
        base_model = model.split("@")[0] if "@" in model else model
        if base_model in ANTHROPIC_STRUCTURED_OUTPUT_MODELS:
            return True
        return any(base_model.startswith(prefix) for prefix in ANTHROPIC_STRUCTURED_OUTPUT_PREFIXES)

    def _model_supports_structured_output(self, model: str) -> bool:
        """Check if a Gemini model supports structured output with response_schema."""
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
        """Perform a health check on Vertex AI.

        Checks the appropriate backend based on the configured default model:
        - Claude models: Checks anthropic[vertex] SDK
        - Gemini models: Checks google-genai SDK
        """
        start_time = time.time()
        model = self._default_model

        if self._is_claude_model(model):
            return await self._doctor_claude(start_time)
        else:
            return await self._doctor_gemini(start_time)

    async def _doctor_gemini(self, start_time: float) -> DoctorResult:
        """Health check for Gemini on Vertex AI."""
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
            client = self._get_gemini_client()
            # List models to verify credentials work
            list(client.models.list())
            latency_ms = (time.time() - start_time) * 1000

            return DoctorResult(
                ok=True,
                message=f"Vertex AI Gemini is accessible (project: {self._project}, location: {self._location}, model: {self._default_model})",
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = str(e)

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

    async def _doctor_claude(self, start_time: float) -> DoctorResult:
        """Health check for Claude on Vertex AI."""
        if not self._claude_project:
            return DoctorResult(
                ok=False,
                message="ANTHROPIC_VERTEX_PROJECT_ID environment variable not set",
                details={"error": "missing_project"},
            )

        try:
            from anthropic import AsyncAnthropicVertex
        except ImportError:
            return DoctorResult(
                ok=False,
                message="anthropic[vertex] package not installed. Run: pip install anthropic[vertex]",
                details={"error": "missing_package"},
            )

        try:
            client = self._get_claude_client()
            # Minimal API call to verify credentials
            await client.messages.create(
                model=self._default_model,
                max_tokens=1,
                messages=[{"role": "user", "content": "Hi"}],
            )
            latency_ms = (time.time() - start_time) * 1000

            return DoctorResult(
                ok=True,
                message=f"Vertex AI Claude is accessible (project: {self._claude_project}, region: {self._claude_region}, model: {self._default_model})",
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = str(e)

            if "credentials" in error_msg.lower() or "authentication" in error_msg.lower():
                help_msg = (
                    "Authentication failed. Either:\n"
                    "  1. Run 'gcloud auth application-default login', or\n"
                    "  2. Set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON file"
                )
            elif "permission" in error_msg.lower():
                help_msg = f"Permission denied. Ensure the credentials have Vertex AI access in project '{self._claude_project}'"
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
