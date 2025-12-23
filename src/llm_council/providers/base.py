"""Base provider adapter definitions for llm-council."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ErrorType(str, Enum):
    """Classification of provider invocation errors.

    Used to determine retry behavior and provide actionable error messages.
    Non-retryable errors (BILLING, AUTH, CLI_NOT_FOUND) should fail fast.
    """

    NONE = "none"
    TIMEOUT = "timeout"
    CLI_NOT_FOUND = "cli_not_found"
    BILLING = "billing"  # Credits exhausted, payment required
    RATE_LIMIT = "rate_limit"  # Too many requests (429)
    AUTH = "auth"  # API key invalid or missing
    MODEL_UNAVAILABLE = "model_unavailable"
    NETWORK = "network"
    UNKNOWN = "unknown"


# Error types that should NOT be retried (permanent failures)
NON_RETRYABLE_ERRORS = frozenset(
    {
        ErrorType.BILLING,
        ErrorType.AUTH,
        ErrorType.CLI_NOT_FOUND,
    }
)

# Patterns to detect specific error types from error messages
_BILLING_PATTERNS = (
    "insufficient_quota",
    "billing",
    "credit",
    "payment",
    "exceeded your current quota",
    "insufficient credits",
    "account has been suspended",
    "payment required",
    "plan does not include",
    "upgrade your plan",
)

_RATE_LIMIT_PATTERNS = (
    "rate_limit",
    "rate limit",
    "too many requests",
    "429",
    "throttl",
)

_AUTH_PATTERNS = (
    "invalid_api_key",
    "invalid api key",
    "unauthorized",
    "authentication",
    "api key not found",
    "invalid_request_error",
    "401",
)

_MODEL_UNAVAILABLE_PATTERNS = (
    "model not found",
    "model_not_found",
    "does not exist",
    "model is currently overloaded",
    "capacity",
)

_NETWORK_PATTERNS = (
    "connection",
    "network",
    "dns",
    "socket",
    "econnrefused",
    "econnreset",
    "etimedout",
)


def classify_error(error_text: str, return_code: int = -1) -> ErrorType:
    """Classify an error based on error text and return code.

    Args:
        error_text: Error message or stderr output
        return_code: Process return code (0 = success)

    Returns:
        ErrorType classification for the error
    """
    if not error_text and return_code == 0:
        return ErrorType.NONE

    error_lower = error_text.lower() if error_text else ""

    # Check billing errors first (most critical, don't waste money retrying)
    for pattern in _BILLING_PATTERNS:
        if pattern.lower() in error_lower:
            return ErrorType.BILLING

    # Check rate limiting (retryable with backoff)
    for pattern in _RATE_LIMIT_PATTERNS:
        if pattern.lower() in error_lower:
            return ErrorType.RATE_LIMIT

    # Check auth errors (permanent, fix API key)
    for pattern in _AUTH_PATTERNS:
        if pattern.lower() in error_lower:
            return ErrorType.AUTH

    # Check model availability
    for pattern in _MODEL_UNAVAILABLE_PATTERNS:
        if pattern.lower() in error_lower:
            return ErrorType.MODEL_UNAVAILABLE

    # Check for network issues (retryable)
    for pattern in _NETWORK_PATTERNS:
        if pattern in error_lower:
            return ErrorType.NETWORK

    return ErrorType.UNKNOWN


def get_billing_help_url(provider: str) -> str:
    """Get the billing help URL for a provider.

    Args:
        provider: Provider name (e.g., "openai", "anthropic", "google")

    Returns:
        URL to the provider's billing page
    """
    urls = {
        "openai": "https://platform.openai.com/account/billing",
        "codex": "https://platform.openai.com/account/billing",
        "codex-cli": "https://platform.openai.com/account/billing",
        "anthropic": "https://console.anthropic.com/settings/billing",
        "claude": "https://console.anthropic.com/settings/billing",
        "google": "https://console.cloud.google.com/billing",
        "gemini": "https://console.cloud.google.com/billing",
        "gemini-cli": "https://console.cloud.google.com/billing",
        "openrouter": "https://openrouter.ai/account/credits",
    }
    return urls.get(provider.lower(), "Check your provider's billing page")


class ProviderCapabilities(BaseModel):
    """Capability flags and limits for a provider adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    streaming: bool = Field(default=False, description="Supports server-side streaming responses.")
    tool_use: bool = Field(default=False, description="Supports tool/function calling.")
    structured_output: bool = Field(
        default=False, description="Supports structured response formats like JSON schema."
    )
    multimodal: bool = Field(default=False, description="Supports non-text inputs like images.")
    max_tokens: int | None = Field(
        default=None, description="Maximum tokens per response if enforced by the provider."
    )


ProviderCapabilityName = Literal[
    "streaming", "tool_use", "structured_output", "multimodal", "max_tokens"
]


class Message(BaseModel):
    """Canonical message format used across providers."""

    model_config = ConfigDict(extra="allow", frozen=True)

    role: str = Field(..., description="Message role (e.g. system, user, assistant, tool).")
    content: Any = Field(..., description="Message content, usually string or structured blocks.")
    name: str | None = Field(default=None, description="Optional name for the message author.")


class StructuredOutputConfig(BaseModel):
    """Configuration for structured output (JSON schema) requests.

    This provides a provider-agnostic representation that each provider
    transforms to their native API format:
    - OpenAI/OpenRouter: response_format.json_schema wrapper
    - Anthropic: output_format with beta header
    - Google: generation_config.responseJsonSchema
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    json_schema: Mapping[str, Any] = Field(..., description="The JSON Schema to enforce.")
    name: str = Field(
        default="council_output",
        description="Schema name (required by OpenAI/OpenRouter, ignored by others).",
    )
    strict: bool = Field(
        default=True,
        description="Enforce strict schema adherence where supported.",
    )


class ReasoningConfig(BaseModel):
    """Configuration for reasoning/thinking mode.

    This provides a provider-agnostic representation for extended reasoning
    that each provider transforms to their native API format:
    - OpenAI: reasoning_effort parameter for o-series models
    - Anthropic: thinking block with budget_tokens
    - Google: thinking_config with thinking_level or thinking_budget
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = Field(
        default=False,
        description="Enable reasoning/thinking mode for this request.",
    )
    effort: Literal["low", "medium", "high", "none"] | None = Field(
        default=None,
        description="OpenAI reasoning effort level (low/medium/high, 'none' for GPT-5.2+).",
    )
    budget_tokens: int | None = Field(
        default=None,
        ge=1024,
        le=128000,
        description="Anthropic thinking budget in tokens (1024-128000).",
    )
    thinking_level: Literal["minimal", "low", "medium", "high"] | None = Field(
        default=None,
        description="Google Gemini 3.x thinking level.",
    )


class GenerateRequest(BaseModel):
    """Input parameters for a text generation request."""

    model_config = ConfigDict(extra="allow")

    model: str | None = Field(default=None, description="Provider model identifier.")
    prompt: str | None = Field(default=None, description="Prompt string for simple use cases.")
    messages: Sequence[Message] | None = Field(
        default=None, description="Chat-style message sequence."
    )
    max_tokens: int | None = Field(default=None, description="Maximum tokens to generate.")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: Sequence[str] | None = Field(default=None, description="Stop sequences.")
    stream: bool = Field(default=False, description="Request server-side streaming.")
    tools: Sequence[Mapping[str, Any]] | None = Field(
        default=None, description="Tool definitions for tool-use capable models."
    )
    tool_choice: Any | None = Field(default=None, description="Tool choice preference.")
    response_format: Mapping[str, Any] | None = Field(
        default=None,
        description="Legacy structured output constraints (use structured_output instead).",
    )
    structured_output: StructuredOutputConfig | None = Field(
        default=None,
        description="Structured output configuration. Each provider transforms this to their native format.",
    )
    reasoning: ReasoningConfig | None = Field(
        default=None,
        description="Reasoning/thinking configuration. Each provider transforms this to their native format.",
    )
    metadata: Mapping[str, Any] | None = Field(default=None, description="Arbitrary metadata.")

    @model_validator(mode="after")
    def _validate_input(self) -> GenerateRequest:
        """Ensure at least one input modality (prompt or messages) is provided."""

        if not self.prompt and not self.messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")
        return self


class GenerateResponse(BaseModel):
    """Provider-agnostic response payload for a generation call."""

    model_config = ConfigDict(extra="allow")

    text: str | None = Field(default=None, description="Primary text response.")
    content: Any | None = Field(
        default=None, description="Structured content payload for non-text outputs."
    )
    tool_calls: Any | None = Field(default=None, description="Tool call payloads if applicable.")
    usage: Mapping[str, int] | None = Field(
        default=None, description="Token usage information (prompt/completion/total)."
    )
    model: str | None = Field(default=None, description="Resolved model identifier.")
    finish_reason: str | None = Field(default=None, description="Stop reason.")
    raw: Any | None = Field(default=None, description="Raw provider response for debugging.")


GenerateResult = GenerateResponse | AsyncIterator[GenerateResponse]


class DoctorResult(BaseModel):
    """Health check result for providers."""

    model_config = ConfigDict(extra="allow", frozen=True)

    ok: bool = Field(..., description="Whether the provider is healthy.")
    message: str | None = Field(default=None, description="Optional status message.")
    latency_ms: float | None = Field(
        default=None, description="Measured latency in milliseconds for the health check."
    )
    details: Mapping[str, Any] | None = Field(
        default=None, description="Additional diagnostic details."
    )


class ProviderAdapter(ABC):
    """Abstract base class for LLM provider adapters.

    Implementations should:
    - set :attr:`name` to a stable, unique provider identifier (e.g. "openai", "anthropic")
    - set :attr:`capabilities` to a :class:`ProviderCapabilities` instance
    - implement async :meth:`generate`, async :meth:`supports`, and async :meth:`doctor`

    Capability semantics:
    - Boolean capabilities (e.g. ``streaming``) are supported when the corresponding flag is ``True``.
    - ``max_tokens`` is treated as supported when it is not ``None`` (i.e. the provider exposes a known limit).
    """

    name: ClassVar[str]
    capabilities: ClassVar[ProviderCapabilities]

    @abstractmethod
    async def generate(self, request: GenerateRequest) -> GenerateResult:
        """Generate a response for the given request."""

    @abstractmethod
    async def supports(self, capability: ProviderCapabilityName | str) -> bool:
        """Return True if the provider supports the given capability name.

        Implementations typically check :attr:`capabilities`. For consistent behavior across
        adapters (notably for ``max_tokens``), consider delegating to
        :meth:`supports_capability`.
        """

    @abstractmethod
    async def doctor(self) -> DoctorResult:
        """Perform a provider health check and return the result."""

    @classmethod
    def capability_names(cls) -> Iterable[str]:
        """Return supported capability attribute names."""

        return ProviderCapabilities.model_fields.keys()

    @classmethod
    def supports_capability_name(cls, capability: str) -> bool:
        """Return True if *capability* is a known capability field name."""

        return capability in ProviderCapabilities.model_fields

    @classmethod
    def supports_capability(cls, capability: ProviderCapabilityName | str) -> bool:
        """Return True if *capability* is supported by this adapter's declared capabilities.

        This helper provides consistent semantics across capability types, especially for
        ``max_tokens`` which is a numeric limit rather than a boolean flag.
        """

        capability_name = str(capability)
        if not cls.supports_capability_name(capability_name):
            return False
        if capability_name == "max_tokens":
            return cls.capabilities.max_tokens is not None
        return bool(getattr(cls.capabilities, capability_name, False))
