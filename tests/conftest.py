"""Pytest configuration and shared fixtures for llm-council tests."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
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
from llm_council.storage.artifacts import reset_store

# --------------------------------------------------------------------------
# Home-directory isolation
#
# `monkeypatch.setenv("HOME", ...)` is a POSIX idiom and is INERT on Windows:
# `Path.home()` -> `os.path.expanduser("~")` reads USERPROFILE (then
# HOMEDRIVE+HOMEPATH) and never consults HOME. Tests that relied on it therefore
# escaped their tmp_path sandbox and reached the developer's real home. One of
# them ran `council config --init` and silently overwrote a real
# ~/.config/llm-council/config.yaml; others asserted against the real ledger.
#
# `isolate_home` sets every variable each platform actually reads. It is applied
# to EVERY test by the autouse `_hermetic_home` fixture: opt-in isolation fails
# by forgetting, and on a fresh CI runner the first test that forgot created
# ~/.council/ledger.db. The autouse guard below stays on top and turns any
# remaining escape into a loud failure instead of silent data loss.
# --------------------------------------------------------------------------

# Resolved at import time, BEFORE any test can monkeypatch the environment.
_REAL_HOME = Path(os.path.expanduser("~"))
_REAL_LEDGERS: tuple[Path, ...] = (
    _REAL_HOME / ".council" / "ledger.db",  # the default store
    _REAL_HOME / ".claude" / "council-ledger.db",  # legacy, still selected when alone
)
_PROTECTED_PATHS: tuple[Path, ...] = (
    _REAL_HOME / ".config" / "llm-council" / "config.yaml",
    *_REAL_LEDGERS,
)

_HOME_ENV_VARS = ("HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH")
_COUNCIL_HOME_ENV_VARS = ("COUNCIL_HOME", "COUNCIL_ARTIFACT_DIR", "COUNCIL_DB_PATH")


def isolate_home(monkeypatch: pytest.MonkeyPatch, home: Path | str) -> Path:
    """Point ``Path.home()`` at ``home`` on every platform.

    Use this instead of ``monkeypatch.setenv("HOME", ...)``, which does nothing
    on Windows. Also clears the COUNCIL_* overrides so a variable set in the
    developer's own shell cannot pull the test back out to real storage.
    """

    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("HOME", str(home))  # POSIX
    monkeypatch.setenv("USERPROFILE", str(home))  # Windows: what expanduser reads
    drive, tail = os.path.splitdrive(str(home))
    monkeypatch.setenv("HOMEDRIVE", drive)  # Windows: legacy fallback pair
    monkeypatch.setenv("HOMEPATH", tail or os.sep)

    for leaked in _COUNCIL_HOME_ENV_VARS:
        monkeypatch.delenv(leaked, raising=False)

    return home


@pytest.fixture(autouse=True)
def _hermetic_home(
    _protect_real_council_home: None,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Path]:
    """Give every test its own home, council store and provider-lock directory.

    The home is a fresh directory under pytest's temp root, outside the test's own
    ``tmp_path``: that stays entirely the test's, and tests do create their own
    ``tmp_path / "home"``. A test that needs a particular home calls
    ``isolate_home()`` again, which simply re-points it. ``reset_store()`` runs on
    both sides because ``get_store()`` caches a singleton. Provider locks get a
    per-test directory and their on/off switch is cleared, so neither a developer's
    shell nor a live council run can reach or change what a test sees. Requesting
    the guard makes it wrap this whole lifecycle, teardown included.
    """

    home = isolate_home(monkeypatch, tmp_path_factory.mktemp("home"))
    monkeypatch.setenv("LLM_COUNCIL_LOCK_DIR", str(home / "provider-locks"))
    monkeypatch.delenv("LLM_COUNCIL_DISABLE_PROVIDER_LOCKS", raising=False)
    reset_store()
    yield home
    reset_store()


@pytest.fixture
def isolated_home(_hermetic_home: Path) -> Path:
    """The per-test home that ``Path.home()`` resolves to (the suite-wide default)."""

    return _hermetic_home


def _fingerprint(path: Path) -> tuple[bool, int, int]:
    try:
        stat = path.stat()
    except OSError:
        return (False, 0, 0)
    return (True, stat.st_mtime_ns, stat.st_size)


@pytest.fixture(autouse=True)
def _protect_real_council_home() -> Iterator[None]:
    """Fail any test that writes to the real council config or ledger.

    Detection, not prevention -- but it converts the failure mode from "a
    developer's configuration silently reverts and they find out days later"
    into "the test that did it goes red immediately". Every test already runs in
    its own home (``_hermetic_home``), so this firing means a test re-pointed the
    home itself without ``isolate_home()``. A ledger only counts when it appears or
    disappears: live council runs on the same machine change it at any moment.
    """

    before = {path: _fingerprint(path) for path in _PROTECTED_PATHS}
    yield
    for path, was in before.items():
        now = _fingerprint(path)
        if path in _REAL_LEDGERS and was[0] and now[0]:
            # An existing ledger changing says nothing about this test; creating or
            # deleting one does, and no council run does either.
            continue
        if now != was:
            pytest.fail(
                f"This test modified the real council file {path}.\n"
                "It escaped its sandbox -- almost always because it re-pointed HOME "
                "only, which Windows ignores. Use conftest.isolate_home() so "
                "Path.home() resolves inside a per-test temporary home.",
                pytrace=False,
            )


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
