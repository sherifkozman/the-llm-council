"""Global registry for provider adapters."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from importlib import metadata
from threading import RLock
from typing import Any

from .base import ProviderAdapter

_ENTRY_POINT_GROUP = "llm_council.providers"
_log = logging.getLogger(__name__)


class ProviderRegistry:
    """Singleton registry of provider adapter classes."""

    _instance: ProviderRegistry | None = None
    _instance_lock: RLock = RLock()

    def __new__(cls) -> ProviderRegistry:
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    _initialized: bool = False

    def __init__(self) -> None:
        if self._initialized:
            return
        self._providers: dict[str, type[ProviderAdapter]] = {}
        self._lock: RLock = RLock()
        self._initialized = True
        self._discover_entry_points()

    def register_provider(self, name: str, adapter_class: type[ProviderAdapter]) -> None:
        """Register a provider adapter class under the given name."""

        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("Provider name must be a non-empty string.")
        if not issubclass(adapter_class, ProviderAdapter):
            raise TypeError("adapter_class must subclass ProviderAdapter.")
        with self._lock:
            existing = self._providers.get(normalized)
            if existing is not None and existing is not adapter_class:
                raise ValueError(
                    f"Provider '{normalized}' is already registered to {existing.__name__}."
                )
            self._providers[normalized] = adapter_class

    def get_provider(self, name: str) -> ProviderAdapter:
        """Instantiate and return the provider adapter for the given name."""

        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("Provider name must be a non-empty string.")
        with self._lock:
            adapter_class = self._providers.get(normalized)
        if adapter_class is None:
            available = ", ".join(self.list_providers())
            raise KeyError(f"Provider '{normalized}' is not registered. Available: [{available}]")
        return adapter_class()

    def list_providers(self) -> list[str]:
        """Return a sorted list of registered provider names."""

        with self._lock:
            return sorted(self._providers.keys())

    def _discover_entry_points(self) -> None:
        """Load and register provider adapters from entry points."""

        try:
            entry_points = metadata.entry_points()
        except Exception:  # pragma: no cover - defensive for older importlib-metadata
            _log.debug("Failed to read provider entry points.", exc_info=True)
            return

        group = self._select_entry_points(entry_points, _ENTRY_POINT_GROUP)

        for entry_point in group:
            try:
                adapter_class = entry_point.load()
            except Exception:
                _log.debug(
                    "Failed to load provider entry point '%s'.", entry_point.name, exc_info=True
                )
                continue

            if not isinstance(adapter_class, type):
                _log.debug(
                    "Provider entry point '%s' resolved to non-class %r; skipping.",
                    entry_point.name,
                    adapter_class,
                )
                continue

            if not issubclass(adapter_class, ProviderAdapter):
                _log.debug(
                    "Provider entry point '%s' resolved to %s, not a ProviderAdapter; skipping.",
                    entry_point.name,
                    adapter_class.__name__,
                )
                continue

            try:
                self.register_provider(entry_point.name, adapter_class)
            except Exception:
                _log.debug(
                    "Failed to register provider entry point '%s'.", entry_point.name, exc_info=True
                )
                continue

    @staticmethod
    def _select_entry_points(entry_points: Any, group: str) -> Iterable[Any]:
        """Select entry points for *group* across importlib.metadata variants."""

        select = getattr(entry_points, "select", None)
        if callable(select):
            result: Iterable[Any] = select(group=group)
            return result

        # Compatibility with older `importlib_metadata` styles that return a mapping.
        if isinstance(entry_points, dict):
            result = entry_points.get(group, [])
            return result

        return []


def get_registry() -> ProviderRegistry:
    """Return the global provider registry singleton."""

    return ProviderRegistry()
