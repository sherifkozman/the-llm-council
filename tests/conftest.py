"""Pytest configuration and shared fixtures for llm-council tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import ClassVar

import pytest

from llm_council.providers.base import (
    DoctorResult,
    GenerateRequest,
    GenerateResponse,
    Message,
    ProviderAdapter,
    ProviderCapabilities,
)
from llm_council.providers.registry import ProviderRegistry


class MockProvider(ProviderAdapter):
    """Mock provider for testing."""

    name: ClassVar[str] = "mock"
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        streaming=False,
        tool_use=False,
        structured_output=True,
        multimodal=False,
        max_tokens=4096,
    )

    def __init__(
        self,
        response_text: str = '{"result": "mock response"}',
        should_fail: bool = False,
        latency_ms: float = 10.0,
    ) -> None:
        self._response_text = response_text
        self._should_fail = should_fail
        self._latency_ms = latency_ms
        self._call_count = 0

    async def generate(
        self, request: GenerateRequest
    ) -> GenerateResponse | AsyncIterator[GenerateResponse]:
        """Generate a mock response."""
        self._call_count += 1
        if self._should_fail:
            raise RuntimeError("Mock provider failure")

        return GenerateResponse(
            text=self._response_text,
            content=self._response_text,
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            finish_reason="stop",
        )

    async def supports(self, capability: str) -> bool:
        """Check if capability is supported."""
        if not self.supports_capability_name(capability):
            return False
        return getattr(self.capabilities, capability, False)

    async def doctor(self) -> DoctorResult:
        """Return mock health check."""
        return DoctorResult(
            ok=not self._should_fail,
            message="Mock provider OK" if not self._should_fail else "Mock failure",
            latency_ms=self._latency_ms,
        )

    @property
    def call_count(self) -> int:
        """Get number of generate calls."""
        return self._call_count


class StreamingMockProvider(MockProvider):
    """Mock provider that supports streaming."""

    name: ClassVar[str] = "streaming-mock"
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        streaming=True,
        tool_use=False,
        structured_output=True,
        multimodal=False,
        max_tokens=4096,
    )

    async def generate(
        self, request: GenerateRequest
    ) -> GenerateResponse | AsyncIterator[GenerateResponse]:
        """Generate with optional streaming."""
        self._call_count += 1
        if self._should_fail:
            raise RuntimeError("Streaming mock provider failure")

        if request.stream:
            return self._stream_response()

        return GenerateResponse(
            text=self._response_text,
            content=self._response_text,
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            finish_reason="stop",
        )

    async def _stream_response(self) -> AsyncIterator[GenerateResponse]:
        """Stream response in chunks."""
        chunks = self._response_text.split()
        for i, chunk in enumerate(chunks):
            yield GenerateResponse(
                text=chunk + " ",
                content=chunk + " ",
                usage={"prompt_tokens": 100, "completion_tokens": i + 1}
                if i == len(chunks) - 1
                else None,
            )


@pytest.fixture
def mock_provider() -> MockProvider:
    """Create a basic mock provider."""
    return MockProvider()


@pytest.fixture
def failing_provider() -> MockProvider:
    """Create a mock provider that fails."""
    return MockProvider(should_fail=True)


@pytest.fixture
def streaming_provider() -> StreamingMockProvider:
    """Create a streaming mock provider."""
    return StreamingMockProvider()


@pytest.fixture
def mock_registry() -> ProviderRegistry:
    """Create a fresh registry with mock provider registered."""
    registry = ProviderRegistry()
    registry.register_provider("mock", MockProvider)
    return registry


@pytest.fixture
def sample_request() -> GenerateRequest:
    """Create a sample generate request."""
    return GenerateRequest(
        model="test-model",
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello, world!"),
        ],
        max_tokens=100,
        temperature=0.7,
    )


@pytest.fixture
def valid_json_response() -> str:
    """Return a valid JSON response."""
    return '{"implementation_title": "Test", "summary": "A test implementation for validation purposes.", "files": [{"path": "test.py", "action": "create", "description": "Test file"}], "testing_notes": {"manual_tests": ["Run pytest"]}, "reasoning": "This is a test reasoning that explains the implementation decisions made."}'


@pytest.fixture
def invalid_json_response() -> str:
    """Return an invalid JSON response."""
    return "This is not valid JSON at all."


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
