"""Tests for provider base classes and registry."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_council.providers.anthropic import (
    DEFAULT_MODEL as ANTHROPIC_DEFAULT_MODEL,
)
from llm_council.providers.anthropic import (
    AnthropicProvider,
    _prepare_schema_for_anthropic,
)
from llm_council.providers.base import (
    DoctorResult,
    ErrorType,
    GenerateRequest,
    GenerateResponse,
    Message,
    ProviderCapabilities,
    StructuredOutputConfig,
    classify_error,
)
from llm_council.providers.cli.claude_code import ClaudeCodeCLIProvider
from llm_council.providers.cli.codex import CodexCLIProvider
from llm_council.providers.cli.gemini_cli import GeminiCLIProvider
from llm_council.providers.gemini import (
    DEFAULT_MODEL as GEMINI_DEFAULT_MODEL,
)
from llm_council.providers.gemini import (
    GeminiProvider,
)
from llm_council.providers.openai import (
    DEFAULT_MODEL as OPENAI_DEFAULT_MODEL,
)
from llm_council.providers.openai import (
    OpenAIProvider,
)
from llm_council.providers.openrouter import (
    DEFAULT_MODEL as OPENROUTER_DEFAULT_MODEL,
)
from llm_council.providers.openrouter import (
    OpenRouterProvider,
)
from llm_council.providers.registry import ProviderRegistry, get_registry
from llm_council.providers.vertex import VertexAIProvider


class TestProviderCapabilities:
    """Tests for ProviderCapabilities model."""

    def test_default_capabilities(self):
        """Test default capability values."""
        caps = ProviderCapabilities()
        assert caps.streaming is False
        assert caps.tool_use is False
        assert caps.structured_output is False
        assert caps.multimodal is False
        assert caps.max_tokens is None

    def test_custom_capabilities(self):
        """Test custom capability values."""
        caps = ProviderCapabilities(
            streaming=True,
            tool_use=True,
            structured_output=True,
            multimodal=True,
            max_tokens=8192,
        )
        assert caps.streaming is True
        assert caps.tool_use is True
        assert caps.structured_output is True
        assert caps.multimodal is True
        assert caps.max_tokens == 8192


class TestMessage:
    """Tests for Message model."""

    def test_user_message(self):
        """Test creating a user message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_system_message(self):
        """Test creating a system message."""
        msg = Message(role="system", content="You are helpful.")
        assert msg.role == "system"
        assert msg.content == "You are helpful."

    def test_assistant_message(self):
        """Test creating an assistant message."""
        msg = Message(role="assistant", content="Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"


class TestGenerateRequest:
    """Tests for GenerateRequest model."""

    def test_minimal_request(self):
        """Test creating a minimal request."""
        req = GenerateRequest(prompt="Hello")
        assert req.prompt == "Hello"
        assert req.messages is None
        assert req.model is None

    def test_request_with_messages(self):
        """Test request with messages."""
        req = GenerateRequest(
            messages=[
                Message(role="user", content="Hello"),
            ],
            model="test-model",
            max_tokens=100,
        )
        assert len(req.messages) == 1
        assert req.model == "test-model"
        assert req.max_tokens == 100

    def test_request_with_all_options(self):
        """Test request with all options."""
        req = GenerateRequest(
            model="test-model",
            messages=[Message(role="user", content="Test")],
            prompt="Fallback prompt",
            max_tokens=500,
            temperature=0.5,
            top_p=0.9,
            stop=["END"],
            stream=True,
        )
        assert req.model == "test-model"
        assert req.temperature == 0.5
        assert req.top_p == 0.9
        assert req.stop == ["END"]
        assert req.stream is True


class TestGenerateResponse:
    """Tests for GenerateResponse model."""

    def test_basic_response(self):
        """Test creating a basic response."""
        resp = GenerateResponse(text="Hello!", content="Hello!")
        assert resp.text == "Hello!"
        assert resp.content == "Hello!"

    def test_response_with_usage(self):
        """Test response with usage info."""
        resp = GenerateResponse(
            text="Response",
            content="Response",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            finish_reason="stop",
        )
        assert resp.usage["prompt_tokens"] == 10
        assert resp.usage["completion_tokens"] == 5
        assert resp.finish_reason == "stop"


class TestDoctorResult:
    """Tests for DoctorResult model."""

    def test_healthy_result(self):
        """Test healthy doctor result."""
        result = DoctorResult(ok=True, message="All good", latency_ms=50.0)
        assert result.ok is True
        assert result.message == "All good"
        assert result.latency_ms == 50.0

    def test_unhealthy_result(self):
        """Test unhealthy doctor result."""
        result = DoctorResult(
            ok=False,
            message="Connection failed",
            details={"error": "timeout"},
        )
        assert result.ok is False
        assert result.message == "Connection failed"
        assert result.details["error"] == "timeout"


class TestProviderRegistry:
    """Tests for ProviderRegistry."""

    def test_register_provider(self, mock_provider):
        """Test registering a provider."""
        registry = ProviderRegistry()
        registry.register_provider("test", type(mock_provider))
        assert "test" in registry.list_providers()

    def test_get_provider(self, mock_registry):
        """Test getting a registered provider."""
        provider = mock_registry.get_provider("mock")
        assert provider is not None
        assert provider.name == "mock"

    def test_get_unregistered_provider_raises(self):
        """Test getting unregistered provider raises error."""
        registry = ProviderRegistry()
        with pytest.raises(KeyError, match="not registered"):
            registry.get_provider("nonexistent")

    def test_aliases_resolve_to_cli_providers(self, mock_registry):
        """Friendly aliases should resolve to canonical provider names."""
        assert mock_registry.resolve_name("codex") == "codex"
        assert mock_registry.resolve_name("gemini") == "gemini"
        assert mock_registry.resolve_name("claude") == "claude"
        assert mock_registry.resolve_name("google") == "gemini"
        assert mock_registry.resolve_name("codex-cli") == "codex"
        assert mock_registry.resolve_name("gemini-cli") == "gemini-cli"
        assert mock_registry.resolve_name("claude-code") == "claude"

    def test_list_providers(self, mock_registry):
        """Test listing registered providers."""
        providers = mock_registry.list_providers()
        assert "mock" in providers

    def test_duplicate_registration_with_different_class_raises(self, mock_registry):
        """Test registering same name with different class raises error."""
        # Register a different class with the same name
        from conftest import StreamingMockProvider

        with pytest.raises(ValueError, match="already registered"):
            mock_registry.register_provider("mock", StreamingMockProvider)

    def test_global_registry(self):
        """Test global registry singleton."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2


class TestErrorClassification:
    """Tests for shared provider error classification."""

    def test_classify_timeout_variants(self):
        """Timeout-shaped transport errors should classify as retryable timeouts."""
        assert classify_error("upstream request timed out") == ErrorType.TIMEOUT
        assert classify_error("The operation did not complete (read timed out)") == (
            ErrorType.TIMEOUT
        )
        assert classify_error("504 DEADLINE_EXCEEDED") == ErrorType.TIMEOUT
        assert classify_error("Deadline expired before operation could complete.") == (
            ErrorType.TIMEOUT
        )

    def test_invalid_request_error_is_not_treated_as_auth(self):
        """Generic invalid request errors should not be mislabeled as auth failures."""
        assert classify_error('{"type":"invalid_request_error","message":"bad input"}') == (
            ErrorType.UNKNOWN
        )

    def test_chatgpt_codex_model_mismatch_is_model_unavailable(self):
        """ChatGPT-auth Codex model mismatches should classify as model availability issues."""
        assert classify_error(
            "The 'gpt-5.4-codex' model is not supported when using Codex with a ChatGPT account."
        ) == ErrorType.MODEL_UNAVAILABLE

    def test_vertex_model_not_found_is_model_unavailable(self):
        """Vertex publisher model access failures should classify as model availability issues."""
        assert classify_error(
            "404 NOT_FOUND. Publisher Model was not found or your project does not have access to it."
        ) == ErrorType.MODEL_UNAVAILABLE


class TestAnthropicStructuredSchemaNormalization:
    """Tests for Anthropic structured-output schema preparation."""

    def test_prepare_schema_for_anthropic_strips_meta_and_closes_nested_objects(self):
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "decision_context": {
                    "type": "object",
                    "minProperties": 1,
                    "properties": {
                        "question": {"type": "string"},
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string", "minLength": 1},
                            },
                        },
                    },
                },
                "criteria": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "maxLength": 20},
                            "score": {"type": "number", "minimum": 0, "maximum": 10},
                        },
                    },
                },
            },
        }

        normalized = _prepare_schema_for_anthropic(schema)

        assert "$schema" not in normalized
        assert "minProperties" not in normalized["properties"]["decision_context"]
        assert (
            "minLength"
            not in normalized["properties"]["decision_context"]["properties"]["metadata"][
                "properties"
            ]["source"]
        )
        assert "minItems" not in normalized["properties"]["criteria"]
        assert normalized["additionalProperties"] is False
        assert normalized["properties"]["decision_context"]["additionalProperties"] is False
        assert (
            normalized["properties"]["decision_context"]["properties"]["metadata"][
                "additionalProperties"
            ]
            is False
        )
        assert normalized["properties"]["criteria"]["items"]["additionalProperties"] is False
        assert "maximum" not in normalized["properties"]["criteria"]["items"]["properties"]["score"]

    def test_prepare_schema_for_anthropic_preserves_existing_additional_properties(self):
        schema = {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "payload": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                }
            },
        }

        normalized = _prepare_schema_for_anthropic(schema)

        assert normalized["additionalProperties"] is True
        assert normalized["properties"]["payload"]["additionalProperties"] == {"type": "string"}


class TestAnthropicStructuredOutputFallback:
    """Tests for Anthropic structured-output fallback behavior."""

    @pytest.mark.asyncio
    async def test_generate_retries_without_output_format_on_schema_rejection(self):
        provider = AnthropicProvider(api_key="test-key")
        client = AsyncMock()
        provider._client = client

        structured_error = RuntimeError(
            "output_format.schema: For 'number' type, properties maximum, minimum are not supported"
        )
        beta_response = SimpleNamespace(
            content=[SimpleNamespace(text='{"review_summary":"ok"}')],
            model="claude-opus-4-6",
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )

        client.beta.messages.create = AsyncMock(side_effect=[structured_error])
        client.messages.create = AsyncMock(return_value=beta_response)

        request = GenerateRequest(
            messages=[Message(role="user", content="Return JSON")],
            model="claude-opus-4-6",
            structured_output=StructuredOutputConfig(
                json_schema={
                    "type": "object",
                    "properties": {
                        "score": {"type": "number", "minimum": 0, "maximum": 10},
                    },
                    "required": ["score"],
                    "additionalProperties": False,
                }
            ),
        )

        response = await provider.generate(request)

        assert response.text == '{"review_summary":"ok"}'
        assert client.beta.messages.create.await_count == 1
        assert client.messages.create.await_count == 1
        first_kwargs = client.beta.messages.create.await_args_list[0].kwargs
        second_kwargs = client.messages.create.await_args_list[0].kwargs
        assert "output_format" in first_kwargs
        assert "output_format" not in second_kwargs


class TestMockProvider:
    """Tests for mock provider functionality."""

    @pytest.mark.asyncio
    async def test_generate(self, mock_provider, sample_request):
        """Test mock provider generate."""
        response = await mock_provider.generate(sample_request)
        assert response.text is not None
        assert mock_provider.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_failure(self, failing_provider, sample_request):
        """Test mock provider failure."""
        with pytest.raises(RuntimeError, match="Mock provider failure"):
            await failing_provider.generate(sample_request)

    @pytest.mark.asyncio
    async def test_supports_capability(self, mock_provider):
        """Test capability check."""
        assert await mock_provider.supports("structured_output") is True
        assert await mock_provider.supports("streaming") is False

    @pytest.mark.asyncio
    async def test_doctor_healthy(self, mock_provider):
        """Test doctor for healthy provider."""
        result = await mock_provider.doctor()
        assert result.ok is True

    @pytest.mark.asyncio
    async def test_doctor_unhealthy(self, failing_provider):
        """Test doctor for unhealthy provider."""
        result = await failing_provider.doctor()
        assert result.ok is False


class TestOpenAIProviderEnvModel:
    """Tests for OPENAI_MODEL env var fallback in OpenAIProvider."""

    def test_default_model_used_when_no_arg_no_env(self, monkeypatch):
        """DEFAULT_MODEL is used when neither arg nor env var is set."""
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        provider = OpenAIProvider()
        assert provider._default_model == OPENAI_DEFAULT_MODEL

    def test_env_var_used_when_no_arg(self, monkeypatch):
        """OPENAI_MODEL env var is used when no default_model arg is passed."""
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
        provider = OpenAIProvider()
        assert provider._default_model == "gpt-4o"

    def test_arg_takes_precedence_over_env_var(self, monkeypatch):
        """default_model arg overrides OPENAI_MODEL env var."""
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
        provider = OpenAIProvider(default_model="gpt-5.1-mini")
        assert provider._default_model == "gpt-5.1-mini"


class TestOpenAIProviderRequestParams:
    """Tests for model-specific OpenAI request parameter handling."""

    @pytest.mark.asyncio
    async def test_o_series_omits_temperature(self):
        """o-series reasoning models should not receive temperature."""
        provider = OpenAIProvider(api_key="test-key")

        captured_kwargs: dict[str, object] = {}

        async def create(**kwargs):
            captured_kwargs.update(kwargs)
            message = MagicMock(content='{"ok": true}', tool_calls=None)
            choice = MagicMock(message=message, finish_reason="stop")
            usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
            return MagicMock(choices=[choice], usage=usage, model="o3-mini")

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=create)
        provider._client = mock_client

        request = GenerateRequest(
            model="o3-mini",
            messages=[Message(role="user", content="test")],
            temperature=0.2,
            max_tokens=100,
        )

        await provider.generate(request)

        assert "temperature" not in captured_kwargs
        assert captured_kwargs["max_completion_tokens"] == 100

    @pytest.mark.asyncio
    async def test_request_timeout_is_forwarded_to_openai_sdk(self):
        """Per-request timeout should be forwarded to the OpenAI SDK call."""
        provider = OpenAIProvider(api_key="test-key")

        captured_kwargs: dict[str, object] = {}

        async def create(**kwargs):
            captured_kwargs.update(kwargs)
            message = MagicMock(content='{"ok": true}', tool_calls=None)
            choice = MagicMock(message=message, finish_reason="stop")
            usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
            return MagicMock(choices=[choice], usage=usage, model="gpt-5.4")

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=create)
        provider._client = mock_client

        request = GenerateRequest(
            model="gpt-5.4",
            messages=[Message(role="user", content="test")],
            timeout_seconds=19,
        )

        await provider.generate(request)

        assert captured_kwargs["timeout"] == 19


class TestProviderTransportDefaults:
    """Tests for provider-native timeout and retry configuration."""

    def test_openai_client_disables_sdk_retries_by_default(self, monkeypatch):
        """OpenAI should use explicit transport timeout and no SDK retries."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        openai_module = ModuleType("openai")
        mock_client = MagicMock()
        openai_module.AsyncOpenAI = mock_client

        with patch.dict(sys.modules, {"openai": openai_module}):
            provider = OpenAIProvider()
            provider._get_client()

        kwargs = mock_client.call_args.kwargs
        assert kwargs["timeout"] == 15.0
        assert kwargs["max_retries"] == 0

    def test_gemini_client_uses_bounded_http_options(self, monkeypatch):
        """Gemini provider should clamp SDK timeout and retry attempts."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        google_module = ModuleType("google")
        genai_module = ModuleType("google.genai")
        mock_client = MagicMock()
        genai_module.Client = mock_client
        genai_module.types = SimpleNamespace(
            HttpOptions=lambda **kwargs: SimpleNamespace(**kwargs),
            HttpRetryOptions=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        google_module.genai = genai_module

        with patch.dict(sys.modules, {"google": google_module, "google.genai": genai_module}):
            provider = GeminiProvider()
            provider._get_client()

        http_options = mock_client.call_args.kwargs["http_options"]
        assert http_options.timeout == 15000
        assert http_options.retry_options.attempts == 1

    def test_vertex_gemini_client_uses_bounded_http_options(self, monkeypatch):
        """Vertex Gemini client should use the same bounded HTTP settings."""
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")

        google_module = ModuleType("google")
        genai_module = ModuleType("google.genai")
        mock_client = MagicMock()
        genai_module.Client = mock_client
        genai_module.types = SimpleNamespace(
            HttpOptions=lambda **kwargs: SimpleNamespace(**kwargs),
            HttpRetryOptions=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        google_module.genai = genai_module

        with patch.dict(sys.modules, {"google": google_module, "google.genai": genai_module}):
            provider = VertexAIProvider()
            provider._get_gemini_client()

        kwargs = mock_client.call_args.kwargs
        http_options = kwargs["http_options"]
        assert kwargs["vertexai"] is True
        assert http_options.timeout == 15000
        assert http_options.retry_options.attempts == 1


class TestAnthropicProviderEnvModel:
    """Tests for ANTHROPIC_MODEL env var fallback in AnthropicProvider."""

    def test_default_model_used_when_no_arg_no_env(self, monkeypatch):
        """DEFAULT_MODEL is used when neither arg nor env var is set."""
        monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
        provider = AnthropicProvider()
        assert provider._default_model == ANTHROPIC_DEFAULT_MODEL

    def test_env_var_used_when_no_arg(self, monkeypatch):
        """ANTHROPIC_MODEL env var is used when no default_model arg is passed."""
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
        provider = AnthropicProvider()
        assert provider._default_model == "claude-haiku-4-5"

    def test_arg_takes_precedence_over_env_var(self, monkeypatch):
        """default_model arg overrides ANTHROPIC_MODEL env var."""
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
        provider = AnthropicProvider(default_model="claude-sonnet-4-5")
        assert provider._default_model == "claude-sonnet-4-5"


class TestGeminiProviderEnvModel:
    """Tests for GEMINI_MODEL env var fallback in GeminiProvider."""

    def test_default_model_used_when_no_arg_no_env(self, monkeypatch):
        """DEFAULT_MODEL is used when neither arg nor env var is set."""
        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        monkeypatch.delenv("GOOGLE_MODEL", raising=False)
        provider = GeminiProvider()
        assert provider._default_model == GEMINI_DEFAULT_MODEL

    def test_env_var_used_when_no_arg(self, monkeypatch):
        """GEMINI_MODEL env var is used when no default_model arg is passed."""
        monkeypatch.setenv("GEMINI_MODEL", "gemini-3-flash-preview")
        provider = GeminiProvider()
        assert provider._default_model == "gemini-3-flash-preview"

    def test_legacy_google_model_env_still_works(self, monkeypatch):
        """Legacy GOOGLE_MODEL env var is still accepted for compatibility."""
        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        monkeypatch.setenv("GOOGLE_MODEL", "gemini-3-flash-preview")
        provider = GeminiProvider()
        assert provider._default_model == "gemini-3-flash-preview"

    def test_arg_takes_precedence_over_env_var(self, monkeypatch):
        """default_model arg overrides GEMINI_MODEL env var."""
        monkeypatch.setenv("GEMINI_MODEL", "gemini-3-flash-preview")
        monkeypatch.setenv("GOOGLE_MODEL", "gemini-3-flash-preview")
        provider = GeminiProvider(default_model="gemini-3.1-pro-preview")
        assert provider._default_model == "gemini-3.1-pro-preview"


class TestGeminiCLIProviderDoctor:
    """Tests for Gemini CLI doctor auth checks."""

    @pytest.mark.asyncio
    async def test_google_api_key_is_enough_for_doctor(self, monkeypatch):
        """Gemini CLI doctor should accept GOOGLE_API_KEY, not only GEMINI_API_KEY."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        provider = GeminiCLIProvider(cli_path="/opt/homebrew/bin/gemini")

        result = await provider.doctor()

        assert result.ok is True

    @pytest.mark.asyncio
    async def test_vertex_auth_settings_are_enough_for_doctor(self, monkeypatch):
        """Gemini CLI doctor should honor selected vertex-ai auth without API keys."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "the-jarvis-brain")
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
        provider = GeminiCLIProvider(cli_path="/opt/homebrew/bin/gemini")

        with patch.object(
            GeminiCLIProvider,
            "_load_existing_auth_settings",
            return_value={"selectedType": "vertex-ai"},
        ):
            result = await provider.doctor()

        assert result.ok is True
        assert "vertex-ai" in result.message

    def test_legacy_gemini_approval_modes_are_normalized(self):
        """Legacy Gemini approval aliases should map to current CLI values."""
        assert GeminiCLIProvider(cli_path="/opt/homebrew/bin/gemini")._approval_mode == "default"
        assert (
            GeminiCLIProvider(
                cli_path="/opt/homebrew/bin/gemini", approval_mode="confirm"
            )._approval_mode
            == "default"
        )
        assert (
            GeminiCLIProvider(
                cli_path="/opt/homebrew/bin/gemini", approval_mode="auto"
            )._approval_mode
            == "yolo"
        )

    def test_invalid_gemini_approval_mode_raises(self):
        """Unknown Gemini approval modes should fail fast."""
        with pytest.raises(ValueError, match="Unsupported Gemini approval mode"):
            GeminiCLIProvider(cli_path="/opt/homebrew/bin/gemini", approval_mode="invalid")


class TestCodexCLIProviderDoctor:
    """Tests for Codex CLI doctor readiness checks."""

    @pytest.mark.asyncio
    async def test_codex_doctor_uses_login_status(self):
        provider = CodexCLIProvider(cli_path="/usr/local/bin/codex")
        process = AsyncMock()
        process.communicate.return_value = (b"Logged in using ChatGPT\n", b"")
        process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=process) as mock_exec:
            result = await provider.doctor()

        assert result.ok is True
        assert "Logged in using ChatGPT" in result.message
        mock_exec.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_codex_doctor_reports_not_logged_in(self):
        provider = CodexCLIProvider(cli_path="/usr/local/bin/codex")
        process = AsyncMock()
        process.communicate.return_value = (b"Not logged in\n", b"")
        process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=process):
            result = await provider.doctor()

        assert result.ok is False
        assert "Not logged in" in result.message


class TestCLIProviderTimeouts:
    """Tests for CLI provider request timeout overrides."""

    def test_codex_cli_uses_request_timeout(self):
        provider = CodexCLIProvider(cli_path="/Users/kozman/.bun/bin/codex")
        request = GenerateRequest(prompt="test", timeout_seconds=19)
        assert provider._request_timeout(request) == 19.0

    def test_gemini_cli_uses_request_timeout(self):
        provider = GeminiCLIProvider(cli_path="/opt/homebrew/bin/gemini")
        request = GenerateRequest(prompt="test", timeout_seconds=17)
        assert provider._request_timeout(request) == 17.0

    def test_gemini_cli_defaults_to_supported_fast_model(self):
        provider = GeminiCLIProvider(cli_path="/opt/homebrew/bin/gemini")
        assert provider._default_model == "gemini-3-flash-preview"

    def test_claude_code_cli_uses_request_timeout(self):
        provider = ClaudeCodeCLIProvider(cli_path="/Users/kozman/.local/bin/claude")
        request = GenerateRequest(prompt="test", timeout_seconds=23)
        assert provider._request_timeout(request) == 23.0

    @pytest.mark.asyncio
    async def test_gemini_cli_starts_new_session(self):
        provider = GeminiCLIProvider(cli_path="/opt/homebrew/bin/gemini")
        process = AsyncMock()
        process.communicate.return_value = (b'{"response":"ok"}', b"")
        process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=process) as mock_exec:
            response = await provider.generate(GenerateRequest(prompt="test"))

        assert response.text == "ok"
        assert mock_exec.await_args.kwargs["start_new_session"] is True

    @pytest.mark.asyncio
    async def test_codex_cli_starts_new_session(self):
        provider = CodexCLIProvider(cli_path="/usr/local/bin/codex")
        process = AsyncMock()
        process.communicate.return_value = (
            (
                b'{"type":"item.completed","item":{"id":"item_0","type":"agent_message",'
                b'"text":"ok"}}\n'
                b'{"type":"turn.completed","usage":{"input_tokens":5,'
                b'"cached_input_tokens":2,"output_tokens":3}}'
            ),
            b"",
        )
        process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=process) as mock_exec:
            response = await provider.generate(GenerateRequest(prompt="test"))

        assert response.text == "ok"
        assert response.usage == {
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
        }
        assert mock_exec.await_args.kwargs["start_new_session"] is True

    def test_codex_cli_defaults_to_chatgpt_compatible_model(self):
        """Codex CLI should default to a model that works with ChatGPT auth."""
        provider = CodexCLIProvider(cli_path="/usr/local/bin/codex")
        assert provider._default_model == "gpt-5.4"
        assert provider.capabilities.structured_output is True

    @pytest.mark.asyncio
    async def test_codex_cli_rewrites_codex_model_for_chatgpt_auth(self):
        """`*-codex` models should be normalized when local Codex auth uses ChatGPT."""
        provider = CodexCLIProvider(cli_path="/usr/local/bin/codex")

        login_process = AsyncMock()
        login_process.communicate.return_value = (b"Logged in using ChatGPT\n", b"")
        login_process.returncode = 0

        exec_process = AsyncMock()
        exec_process.communicate.return_value = (
            (
                b'{"type":"item.completed","item":{"id":"item_0","type":"agent_message",'
                b'"text":"READY"}}\n'
                b'{"type":"turn.completed","usage":{"input_tokens":11,"output_tokens":1}}'
            ),
            b"",
        )
        exec_process.returncode = 0

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=[login_process, exec_process],
        ) as mock_exec:
            response = await provider.generate(
                GenerateRequest(prompt="test", model="gpt-5.4-codex")
            )

        assert response.text == "READY"
        assert mock_exec.await_args_list[1].args[:4] == (
            "/usr/local/bin/codex",
            "exec",
            "--sandbox",
            "read-only",
        )
        assert "--json" in mock_exec.await_args_list[1].args
        assert "-m" in mock_exec.await_args_list[1].args
        model_index = mock_exec.await_args_list[1].args.index("-m")
        assert mock_exec.await_args_list[1].args[model_index + 1] == "gpt-5.4"

    @pytest.mark.asyncio
    async def test_codex_cli_error_reports_useful_tail(self):
        """Codex CLI errors should retain the meaningful trailing error line."""
        provider = CodexCLIProvider(cli_path="/usr/local/bin/codex")
        process = AsyncMock()
        process.communicate.return_value = (
            b"",
            (
                b"OpenAI Codex v0.117.0 (research preview)\n"
                b"model: gpt-5.4-codex\n"
                b"ERROR: {\"type\":\"error\",\"status\":400,\"error\":{\"type\":"
                b"\"invalid_request_error\",\"message\":\"The 'gpt-5.4-codex' model is not "
                b"supported when using Codex with a ChatGPT account.\"}}\n"
            ),
        )
        process.returncode = 1

        with (
            patch("asyncio.create_subprocess_exec", return_value=process),
            pytest.raises(RuntimeError, match="MODEL UNAVAILABLE: ERROR:"),
        ):
            await provider.generate(GenerateRequest(prompt="test"))

    @pytest.mark.asyncio
    async def test_codex_cli_extracts_jsonl_error_when_stderr_is_empty(self):
        """Codex JSONL errors should still classify correctly when stderr is blank."""
        provider = CodexCLIProvider(cli_path="/usr/local/bin/codex")
        process = AsyncMock()
        process.communicate.return_value = (
            (
                b'{"type":"thread.started","thread_id":"abc"}\n'
                b'{"type":"error","message":"{\\"type\\":\\"error\\",\\"error\\":{'
                b'\\"type\\":\\"invalid_request_error\\",\\"code\\":\\"invalid_json_schema\\",'
                b'\\"message\\":\\"Invalid schema\\"},\\"status\\":400}"}\n'
                b'{"type":"turn.failed","error":{"message":"Invalid schema"}}'
            ),
            b"",
        )
        process.returncode = 1

        with (
            patch("asyncio.create_subprocess_exec", return_value=process),
            pytest.raises(RuntimeError, match="invalid_json_schema|Invalid schema"),
        ):
            await provider.generate(GenerateRequest(prompt="test"))

    @pytest.mark.asyncio
    async def test_claude_cli_starts_new_session(self):
        provider = ClaudeCodeCLIProvider(cli_path="/usr/local/bin/claude")
        process = AsyncMock()
        process.communicate.return_value = (b'{"result": "ok"}', b"")
        process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=process) as mock_exec:
            response = await provider.generate(GenerateRequest(prompt="test"))

        assert response.text == "ok"
        assert mock_exec.await_args.kwargs["start_new_session"] is True

    @pytest.mark.asyncio
    async def test_claude_cli_parses_list_envelopes(self):
        provider = ClaudeCodeCLIProvider(cli_path="/usr/local/bin/claude")
        process = AsyncMock()
        process.communicate.return_value = (
            (
                b'[{"type":"assistant","message":{"content":[{"type":"text","text":"READY"}]}},'
                b'{"type":"result","usage":{"input_tokens":11,"output_tokens":7}}]'
            ),
            b"",
        )
        process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=process):
            response = await provider.generate(GenerateRequest(prompt="test"))

        assert response.text == "READY"
        assert response.usage == {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
        }

    @pytest.mark.asyncio
    async def test_gemini_cli_uses_documented_json_prompt_mode(self):
        provider = GeminiCLIProvider(cli_path="/opt/homebrew/bin/gemini")
        process = AsyncMock()
        process.communicate.return_value = (
            b'{"response":"READY","stats":{"inputTokenCount":5,"outputTokenCount":3}}',
            b"",
        )
        process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=process) as mock_exec:
            response = await provider.generate(GenerateRequest(prompt="test"))

        assert response.text == "READY"
        assert response.usage == {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
        }
        assert mock_exec.await_args.args[:5] == (
            "/opt/homebrew/bin/gemini",
            "-p",
            "test",
            "--approval-mode",
            "default",
        )
        assert "-m" in mock_exec.await_args.args
        assert "--output-format" in mock_exec.await_args.args

    @pytest.mark.asyncio
    async def test_gemini_cli_uses_isolated_home_and_preserves_auth_settings(self):
        provider = GeminiCLIProvider(cli_path="/opt/homebrew/bin/gemini")
        process = AsyncMock()
        process.communicate.return_value = (b'{"response":"READY"}', b"")
        process.returncode = 0
        captured_settings: dict[str, object] = {}

        def _fake_exec(*args, **kwargs):
            env = kwargs["env"]
            settings_path = Path(env["GEMINI_CLI_HOME"]) / ".gemini" / "settings.json"
            captured_settings["payload"] = json.loads(settings_path.read_text(encoding="utf-8"))
            return process

        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test-google-key"}, clear=True),
            patch.object(
                GeminiCLIProvider,
                "_load_existing_auth_settings",
                return_value={"selectedType": "vertex-ai"},
            ),
            patch("asyncio.create_subprocess_exec", side_effect=_fake_exec) as mock_exec,
        ):
            response = await provider.generate(GenerateRequest(prompt="test"))

        env = mock_exec.await_args.kwargs["env"]

        assert response.text == "READY"
        assert env["GOOGLE_API_KEY"] == "test-google-key"
        assert "GEMINI_API_KEY" not in env
        assert captured_settings["payload"] == {
            "admin": {
                "extensions": {"enabled": False},
                "mcp": {"enabled": False},
            },
            "security": {
                "auth": {"selectedType": "vertex-ai"},
            },
        }

    @pytest.mark.asyncio
    async def test_gemini_cli_vertex_auth_drops_conflicting_google_api_key_when_project_is_set(self):
        provider = GeminiCLIProvider(cli_path="/opt/homebrew/bin/gemini")
        process = AsyncMock()
        process.communicate.return_value = (b'{"response":"READY"}', b"")
        process.returncode = 0

        with (
            patch.dict(
                "os.environ",
                {
                    "GOOGLE_API_KEY": "test-google-key",
                    "GOOGLE_CLOUD_PROJECT": "the-jarvis-brain",
                    "GOOGLE_CLOUD_LOCATION": "global",
                    "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/adc.json",
                },
                clear=False,
            ),
            patch.object(
                GeminiCLIProvider,
                "_load_existing_auth_settings",
                return_value={"selectedType": "vertex-ai"},
            ),
            patch("asyncio.create_subprocess_exec", return_value=process) as mock_exec,
        ):
            response = await provider.generate(GenerateRequest(prompt="test"))

        env = mock_exec.await_args.kwargs["env"]

        assert response.text == "READY"
        assert env["GOOGLE_CLOUD_PROJECT"] == "the-jarvis-brain"
        assert env["GOOGLE_CLOUD_LOCATION"] == "global"
        assert env["GOOGLE_APPLICATION_CREDENTIALS"] == "/tmp/adc.json"
        assert "GOOGLE_API_KEY" not in env

    @pytest.mark.asyncio
    async def test_gemini_cli_parses_json_after_stdout_warnings(self):
        provider = GeminiCLIProvider(cli_path="/opt/homebrew/bin/gemini")
        process = AsyncMock()
        process.communicate.return_value = (
            (
                b"Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.\n"
                b'{\n  "session_id": "abc",\n  "response": "READY",\n'
                b'  "stats": {"inputTokenCount": 5, "outputTokenCount": 2}\n}\n'
            ),
            b"",
        )
        process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=process):
            response = await provider.generate(GenerateRequest(prompt="test"))

        assert response.text == "READY"
        assert response.usage == {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7,
        }

    @pytest.mark.asyncio
    async def test_codex_cli_reads_last_message_file(self):
        provider = CodexCLIProvider(cli_path="/usr/local/bin/codex")
        process = AsyncMock()
        process.communicate.return_value = (b"", b"")
        process.returncode = 0
        captured_output_path: dict[str, str] = {}

        def _fake_exec(*args, **kwargs):
            output_index = args.index("-o")
            output_path = args[output_index + 1]
            Path(output_path).write_text("READY\n", encoding="utf-8")
            captured_output_path["path"] = output_path
            return process

        with patch("asyncio.create_subprocess_exec", side_effect=_fake_exec) as mock_exec:
            response = await provider.generate(GenerateRequest(prompt="test"))

        assert response.text == "READY\n"
        assert "-o" in mock_exec.await_args.args
        assert not Path(captured_output_path["path"]).exists()

    @pytest.mark.asyncio
    async def test_codex_cli_uses_isolated_home_for_subprocess(self):
        provider = CodexCLIProvider(cli_path="/usr/local/bin/codex")
        process = AsyncMock()
        process.communicate.return_value = (
            b'{"type":"item.completed","item":{"type":"agent_message","text":"READY"}}',
            b"",
        )
        process.returncode = 0

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-openai-key"}, clear=False),
            patch.object(provider, "_create_isolated_cli_home", return_value="/tmp/codex-home"),
            patch("asyncio.create_subprocess_exec", return_value=process) as mock_exec,
        ):
            response = await provider.generate(GenerateRequest(prompt="test"))

        env = mock_exec.await_args.kwargs["env"]

        assert response.text == "READY"
        assert env["HOME"] == "/tmp/codex-home"
        assert env["OPENAI_API_KEY"] == "test-openai-key"

    @pytest.mark.asyncio
    async def test_codex_cli_falls_back_to_jsonl_agent_message_when_output_file_is_empty(self):
        provider = CodexCLIProvider(cli_path="/usr/local/bin/codex")
        process = AsyncMock()
        process.communicate.return_value = (
            (
                b'{"type":"thread.started","thread_id":"abc"}\n'
                b'{"type":"item.completed","item":{"id":"item_0","type":"agent_message",'
                b'"text":"{\\"answer\\":\\"READY\\"}"}}\n'
                b'{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":4}}'
            ),
            b"",
        )
        process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=process):
            response = await provider.generate(GenerateRequest(prompt="test"))

        assert response.text == '{"answer":"READY"}'
        assert response.usage == {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
        }

    @pytest.mark.asyncio
    async def test_codex_cli_ignores_non_json_stdout_before_jsonl_events(self):
        provider = CodexCLIProvider(cli_path="/usr/local/bin/codex")
        process = AsyncMock()
        process.communicate.return_value = (
            (
                b"WARNING: transient banner\n"
                b'{"type":"thread.started","thread_id":"abc"}\n'
                b'{"type":"item.completed","item":{"type":"agent_message","text":"READY"}}\n'
                b'{"type":"turn.completed","usage":{"input_tokens":8,"output_tokens":2}}'
            ),
            b"",
        )
        process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=process):
            response = await provider.generate(GenerateRequest(prompt="test"))

        assert response.text == "READY"
        assert response.usage == {
            "prompt_tokens": 8,
            "completion_tokens": 2,
            "total_tokens": 10,
        }

    def test_codex_cli_isolated_home_tolerates_missing_auth_files(self, tmp_path, monkeypatch):
        source_home = tmp_path / "source-home"
        monkeypatch.setattr("llm_council.providers.cli.codex.Path.home", lambda: source_home)

        provider = CodexCLIProvider(cli_path="/usr/local/bin/codex")
        cli_home = provider._create_isolated_cli_home()

        try:
            isolated_codex_dir = Path(cli_home) / ".codex"
            assert isolated_codex_dir.exists()
            assert list(isolated_codex_dir.iterdir()) == []
        finally:
            shutil.rmtree(cli_home, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_codex_cli_structured_output_normalizes_nested_object_schema(self):
        provider = CodexCLIProvider(cli_path="/usr/local/bin/codex")
        process = AsyncMock()
        process.communicate.return_value = (
            b'{"type":"item.completed","item":{"type":"agent_message","text":"{\\"ok\\":true}"}}',
            b"",
        )
        process.returncode = 0
        captured_schema: dict[str, object] = {}

        def _fake_exec(*args, **kwargs):
            schema_index = args.index("--output-schema")
            schema_path = args[schema_index + 1]
            captured_schema["value"] = json.loads(Path(schema_path).read_text(encoding="utf-8"))
            return process

        request = GenerateRequest(
            prompt="test",
            structured_output=StructuredOutputConfig(
                json_schema={
                    "type": "object",
                    "properties": {
                        "context": {
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                        }
                    },
                },
                name="codex_test",
                strict=True,
            ),
        )

        with patch("asyncio.create_subprocess_exec", side_effect=_fake_exec):
            await provider.generate(request)

        saved_schema = captured_schema["value"]
        assert saved_schema["additionalProperties"] is False
        assert saved_schema["properties"]["context"]["additionalProperties"] is False
        assert saved_schema["properties"]["context"]["required"] == ["value"]

    @pytest.mark.asyncio
    async def test_codex_cli_cleans_up_isolated_home_when_command_build_fails(self, tmp_path):
        provider = CodexCLIProvider(cli_path="/usr/local/bin/codex")
        cli_home = tmp_path / "codex-home"

        with (
            patch.object(provider, "_create_isolated_cli_home", return_value=str(cli_home)),
            patch.object(provider, "_build_command", side_effect=ValueError("bad request")),
            pytest.raises(ValueError, match="bad request"),
        ):
            await provider.generate(GenerateRequest(prompt="test"))

        assert not cli_home.exists()


class TestOpenRouterProviderEnvModel:
    """Tests for OPENROUTER_MODEL env var fallback in OpenRouterProvider."""

    def test_default_model_used_when_no_arg_no_env(self, monkeypatch):
        """DEFAULT_MODEL is used when neither arg nor env var is set."""
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
        provider = OpenRouterProvider()
        assert provider._default_model == OPENROUTER_DEFAULT_MODEL

    def test_env_var_used_when_no_arg(self, monkeypatch):
        """OPENROUTER_MODEL env var is used when no default_model arg is passed."""
        monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-4o")
        provider = OpenRouterProvider()
        assert provider._default_model == "openai/gpt-4o"

    def test_arg_takes_precedence_over_env_var(self, monkeypatch):
        """default_model arg overrides OPENROUTER_MODEL env var."""
        monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-4o")
        provider = OpenRouterProvider(default_model="anthropic/claude-opus-4-5")
        assert provider._default_model == "anthropic/claude-opus-4-5"
