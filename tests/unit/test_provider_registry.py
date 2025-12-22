from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pytest

from llm_council.providers.base import (
    DoctorResult,
    GenerateRequest,
    GenerateResponse,
    ProviderAdapter,
    ProviderCapabilities,
)
from llm_council.providers.registry import ProviderRegistry


class DummyProvider(ProviderAdapter):
    name = "dummy"
    capabilities = ProviderCapabilities()

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        return GenerateResponse(text="ok")

    async def supports(self, capability: str) -> bool:
        return False

    async def doctor(self) -> DoctorResult:
        return DoctorResult(ok=True)


class OtherDummyProvider(DummyProvider):
    name = "dummy2"


def _reset_registry_singleton() -> None:
    ProviderRegistry._instance = None


def test_registry_is_singleton() -> None:
    _reset_registry_singleton()
    r1 = ProviderRegistry()
    r2 = ProviderRegistry()
    assert r1 is r2


def test_register_and_get_provider_instance() -> None:
    _reset_registry_singleton()
    registry = ProviderRegistry()

    registry.register_provider("dummy", DummyProvider)
    provider = registry.get_provider("dummy")
    assert isinstance(provider, DummyProvider)


def test_register_duplicate_name_rejected() -> None:
    _reset_registry_singleton()
    registry = ProviderRegistry()

    registry.register_provider("dummy", DummyProvider)
    with pytest.raises(ValueError):
        registry.register_provider("dummy", OtherDummyProvider)


@dataclass(frozen=True)
class _FakeEntryPoint:
    name: str
    value: Any

    def load(self) -> Any:
        return self.value


class _FakeEntryPoints:
    def __init__(self, items: Iterable[_FakeEntryPoint]) -> None:
        self._items = list(items)

    def select(self, *, group: str) -> list[_FakeEntryPoint]:
        # The registry uses `select(group=...)`; we don't filter by group in this stub.
        return list(self._items)


def test_entry_point_auto_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_registry_singleton()

    from llm_council.providers import registry as registry_module

    def fake_entry_points() -> _FakeEntryPoints:
        return _FakeEntryPoints([_FakeEntryPoint(name="dummy", value=DummyProvider)])

    monkeypatch.setattr(registry_module.metadata, "entry_points", fake_entry_points)

    registry = ProviderRegistry()
    assert "dummy" in registry.list_providers()
