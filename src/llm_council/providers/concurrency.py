"""Cross-process coordination helpers for provider calls.

These guards are intentionally lightweight. They serialize live requests to the
same provider across separate ``council`` processes so one client session cannot
stampede a provider with overlapping doctor probes, planner runs, and review
runs.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

try:  # pragma: no cover - exercised indirectly on non-POSIX platforms
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None

_LOCK_DIR_ENV = "LLM_COUNCIL_LOCK_DIR"
_DISABLE_LOCKS_ENV = "LLM_COUNCIL_DISABLE_PROVIDER_LOCKS"
_DEFAULT_POLL_INTERVAL_SECONDS = 0.05


def _locks_enabled() -> bool:
    raw = os.getenv(_DISABLE_LOCKS_ENV, "").strip().lower()
    return fcntl is not None and raw not in {"1", "true", "yes", "on"}


def _lock_root() -> Path:
    override = os.getenv(_LOCK_DIR_ENV)
    base = Path(override) if override else Path(tempfile.gettempdir()) / "llm-council-provider-locks"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _lock_path_for_provider(provider_name: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in provider_name)
    return _lock_root() / f"{safe_name}.lock"


@dataclass
class ProviderCallLease:
    """Represents an acquired provider slot."""

    provider_name: str
    lock_path: Path | None
    fd: int | None
    wait_ms: float = 0.0

    def release(self) -> None:
        """Release the provider slot."""
        if self.fd is None or fcntl is None:
            return
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.fd = None


def acquire_provider_call_lease(
    provider_name: str,
    *,
    timeout_seconds: float | None = None,
    poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
) -> ProviderCallLease:
    """Acquire a cross-process exclusive lease for a provider."""

    if not _locks_enabled():
        return ProviderCallLease(provider_name=provider_name, lock_path=None, fd=None, wait_ms=0.0)

    lock_path = _lock_path_for_provider(provider_name)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    started = time.monotonic()

    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError as exc:
            elapsed = time.monotonic() - started
            if timeout_seconds is not None and elapsed >= timeout_seconds:
                os.close(fd)
                raise TimeoutError(
                    f"Timed out waiting for provider slot: {provider_name}"
                ) from exc
            time.sleep(poll_interval_seconds)

    waited_ms = round((time.monotonic() - started) * 1000, 1)
    return ProviderCallLease(
        provider_name=provider_name,
        lock_path=lock_path,
        fd=fd,
        wait_ms=waited_ms,
    )


@asynccontextmanager
async def provider_call_slot(
    provider_name: str,
    *,
    timeout_seconds: float | None = None,
    poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
) -> AsyncIterator[float]:
    """Async wrapper around :func:`acquire_provider_call_lease`."""

    lease = await asyncio.to_thread(
        acquire_provider_call_lease,
        provider_name,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    try:
        yield lease.wait_ms
    finally:
        await asyncio.to_thread(lease.release)
