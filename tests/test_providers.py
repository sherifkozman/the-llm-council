"""Tests for provider base classes and registry."""

from __future__ import annotations

import pytest

from llm_council.providers.base import (
    DoctorResult,
    GenerateRequest,
    GenerateResponse,
    Message,
    ProviderCapabilities,
)
from llm_council.providers.registry import ProviderRegistry, get_registry


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
